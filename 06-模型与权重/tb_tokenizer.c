#include "tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

int main(int argc, char** argv) {
    printf("====================================================\n");
    printf(" Testing 120-line C BPE Tokenizer for Zynq-7020 ARM\n");
    printf("====================================================\n");

    const char* path = (argc > 1) ? argv[1] : "/mnt/qwen/tokenizer_64k.bin";
    Tokenizer tok;
    int rc = tokenizer_init(&tok, path);
    printf("[*] tokenizer_init returned: %d (0 = SUCCESS)\n", rc);
    assert(rc == 0);
    printf("[*] Loaded %u tokens, max_len=%u\n", tok.vocab_size, tok.max_token_len);

    const char* test_prompts[] = {
        "Hello, world!",
        "2026 FPGA Contest",
        "Qwen3.5 0.8B Linear Attention",
        "{\"action\": \"think\", \"step\": 1}"
    };

    int tokens[256];
    char decoded[1024];

    for (int t = 0; t < 4; t++) {
        const char* prompt = test_prompts[t];
        int n = tokenizer_encode(&tok, prompt, tokens, 256);
        printf("\n[PROMPT %d] \"%s\"\n", t + 1, prompt);
        printf("  Encoded into %d tokens: [", n);
        for (int i = 0; i < n; i++) {
            printf("%d%s", tokens[i], (i == n - 1) ? "" : ", ");
        }
        printf("]\n");

        // Decode back
        decoded[0] = '\0';
        char piece[256];
        for (int i = 0; i < n; i++) {
            tokenizer_decode(&tok, tokens[i], piece, sizeof(piece));
            strcat(decoded, piece);
        }
        printf("  Decoded text: \"%s\"\n", decoded);
        assert(strcmp(prompt, decoded) == 0);
        printf("  [PASS] 100%% Exact Roundtrip Match!\n");
    }

    tokenizer_free(&tok);
    printf("\n====================================================\n");
    printf(" ALL TOKENIZER TESTS PASSED (100%% EXACT C99 MATCH)\n");
    printf("====================================================\n");
    return 0;
}

