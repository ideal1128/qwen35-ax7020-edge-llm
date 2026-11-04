#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#define VOCAB_SIZE 65536
#define HASH_TABLE_SIZE 131072
#define MAX_TOKEN_STR_LEN 256

typedef struct {
    uint32_t str_offset; // offset in str_pool
    uint8_t  len;
    int32_t  rank;
} TokenEntry;

typedef struct {
    uint32_t key_offset; // offset in str_pool
    uint8_t  key_len;
    int32_t  token_id;
} HashNode;

typedef struct {
    uint32_t vocab_size;
    uint32_t max_token_len;
    TokenEntry entries[VOCAB_SIZE];
    char* str_pool;
    size_t pool_size;
    HashNode hash_table[HASH_TABLE_SIZE];
} Tokenizer;

// Initialize tokenizer from binary file (e.g. tokenizer_64k.bin or tokenizer.bin)
int tokenizer_init(Tokenizer* tok, const char* bin_path);

// Free tokenizer memory
void tokenizer_free(Tokenizer* tok);

// Allocate and initialize on heap
Tokenizer* tokenizer_create(const char* bin_path);
void tokenizer_destroy(Tokenizer* tok);

// Encode text into token IDs
int tokenizer_encode(const Tokenizer* tok, const char* text, int* tokens, int max_tokens);

// Decode single token ID into output buffer
int tokenizer_decode(const Tokenizer* tok, int token_id, char* buf, int max_buf_len);

#endif // TOKENIZER_H

