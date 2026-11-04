#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t hash_bytes(const char* data, uint8_t len) {
    uint32_t h = 2166136261u;
    for (uint8_t i = 0; i < len; i++) {
        h ^= (uint8_t)data[i];
        h *= 16777619u;
    }
    return h;
}

int tokenizer_init(Tokenizer* tok, const char* bin_path) {
    memset(tok, 0, sizeof(Tokenizer));
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        tok->hash_table[i].token_id = -1;
    }

    FILE* f = fopen(bin_path, "rb");
    if (!f) return -1;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4) {
        fclose(f);
        return -2;
    }
    if (memcmp(magic, "TK32", 4) != 0 && memcmp(magic, "TK64", 4) != 0) {
        fclose(f);
        return -2;
    }

    uint32_t count = 0, max_len = 0;
    if (fread(&count, 4, 1, f) != 1 || fread(&max_len, 4, 1, f) != 1 || count > VOCAB_SIZE) {
        fclose(f);
        return -3;
    }
    tok->vocab_size = count;
    tok->max_token_len = max_len;

    // Allocate memory pool for strings (1MB pool for 64K token vocabulary)
    tok->str_pool = (char*)malloc(1024 * 1024);
    if (!tok->str_pool) {
        fclose(f);
        return -4;
    }
    tok->pool_size = 0;

    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        int32_t rank = 0;
        uint8_t slen = 0;
        if (fread(&rank, 4, 1, f) != 1 || fread(&slen, 1, 1, f) != 1) {
            fclose(f);
            return -5;
        }

        tok->entries[i].str_offset = (uint32_t)tok->pool_size;
        tok->entries[i].len = slen;
        tok->entries[i].rank = rank;

        if (slen > 0) {
            if (fread(tok->str_pool + tok->pool_size, 1, slen, f) != slen) {
                fclose(f);
                return -6;
            }

            // Insert into hash table
            uint32_t h = hash_bytes(tok->str_pool + tok->pool_size, slen) % HASH_TABLE_SIZE;
            while (tok->hash_table[h].token_id != -1) {
                h = (h + 1) % HASH_TABLE_SIZE;
            }
            tok->hash_table[h].key_offset = (uint32_t)tok->pool_size;
            tok->hash_table[h].key_len = slen;
            tok->hash_table[h].token_id = (int32_t)i;

            tok->pool_size += slen;
        }
    }

    fclose(f);
    return 0;
}

void tokenizer_free(Tokenizer* tok) {
    if (tok && tok->str_pool) {
        free(tok->str_pool);
        tok->str_pool = NULL;
    }
}

Tokenizer* tokenizer_create(const char* bin_path) {
    Tokenizer* tok = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    if (!tok) return NULL;
    int rc = tokenizer_init(tok, bin_path);
    if (rc != 0) {
        free(tok);
        return NULL;
    }
    return tok;
}

void tokenizer_destroy(Tokenizer* tok) {
    if (tok) {
        tokenizer_free(tok);
        free(tok);
    }
}

static int lookup_token(const Tokenizer* tok, const char* str, uint8_t len) {
    uint32_t h = hash_bytes(str, len) % HASH_TABLE_SIZE;
    while (tok->hash_table[h].token_id != -1) {
        if (tok->hash_table[h].key_len == len &&
            memcmp(tok->str_pool + tok->hash_table[h].key_offset, str, len) == 0) {
            return tok->hash_table[h].token_id;
        }
        h = (h + 1) % HASH_TABLE_SIZE;
    }
    return -1;
}

int tokenizer_encode(const Tokenizer* tok, const char* text, int* tokens, int max_tokens) {
    if (!text || !tokens || max_tokens <= 0) return 0;

    int n_tokens = 0;
    size_t text_len = strlen(text);

    // Initial pass: lookup individual UTF-8 characters / single bytes
    size_t i = 0;
    while (i < text_len && n_tokens < max_tokens) {
        // Try up to 4-byte UTF-8 character
        int char_len = 1;
        unsigned char c = (unsigned char)text[i];
        if ((c & 0x80) == 0x00) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        if (i + char_len > text_len) char_len = (int)(text_len - i);

        int id = lookup_token(tok, text + i, (uint8_t)char_len);
        if (id == -1) {
            // Fallback: single byte lookup
            id = lookup_token(tok, text + i, 1);
            char_len = 1;
        }
        if (id != -1) {
            tokens[n_tokens++] = id;
        }
        i += char_len;
    }

    // Iterative BPE Merge loop
    while (1) {
        int best_idx = -1;
        int best_merged_id = -1;
        int32_t best_rank = 0x7FFFFFFF;

        for (int p = 0; p < n_tokens - 1; p++) {
            int id1 = tokens[p];
            int id2 = tokens[p + 1];
            uint8_t l1 = tok->entries[id1].len;
            uint8_t l2 = tok->entries[id2].len;
            if (l1 + l2 >= MAX_TOKEN_STR_LEN) continue;

            char merged_str[MAX_TOKEN_STR_LEN];
            memcpy(merged_str, tok->str_pool + tok->entries[id1].str_offset, l1);
            memcpy(merged_str + l1, tok->str_pool + tok->entries[id2].str_offset, l2);

            int merged_id = lookup_token(tok, merged_str, l1 + l2);
            if (merged_id != -1) {
                int32_t rank = tok->entries[merged_id].rank;
                if (rank < best_rank) {
                    best_rank = rank;
                    best_idx = p;
                    best_merged_id = merged_id;
                }
            }
        }

        if (best_idx == -1) break; // No more merges possible

        // Apply merge
        tokens[best_idx] = best_merged_id;
        for (int p = best_idx + 1; p < n_tokens - 1; p++) {
            tokens[p] = tokens[p + 1];
        }
        n_tokens--;
    }

    return n_tokens;
}

int tokenizer_decode(const Tokenizer* tok, int token_id, char* buf, int max_buf_len) {
    if (!tok || token_id < 0 || token_id >= (int)tok->vocab_size || !buf || max_buf_len <= 0) {
        return 0;
    }
    uint8_t len = tok->entries[token_id].len;
    if (len >= max_buf_len) len = max_buf_len - 1;
    memcpy(buf, tok->str_pool + tok->entries[token_id].str_offset, len);
    buf[len] = '\0';
    return len;
}

