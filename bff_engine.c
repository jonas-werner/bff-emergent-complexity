/*
 * BFF interpreter + batch epoch runner in C with OpenMP.
 * Compiled as shared lib, called from Python via ctypes.
 *
 * Each cell is (uint8_t value, int64_t token_id).
 * The soup is a flat array: soup[pop_size * prog_len] of cells.
 * One interaction: pick two programs, concatenate into 128-byte tape,
 * run BFF, split back. One epoch = pop_size interactions.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define PROG_LEN 64
#define TAPE_LEN 128  /* 2 * PROG_LEN */
#define TAPE_MOD 256
#define DEFAULT_MAX_STEPS 16384

typedef struct {
    uint8_t value;
    int64_t token_id;
} Cell;

/* Run BFF on tape of length `n`. Returns steps executed. */
static int run_bff(Cell *tape, int n, int max_steps) {
    int ip = 0, rh = 0, wh = 0, steps = 0;

    while (ip >= 0 && ip < n && steps < max_steps) {
        steps++;
        uint8_t cmd = tape[ip].value;

        switch (cmd) {
        case '<': rh = (rh - 1 + n) % n; break;
        case '>': rh = (rh + 1) % n; break;
        case '{': wh = (wh - 1 + n) % n; break;
        case '}': wh = (wh + 1) % n; break;
        case '-': tape[rh].value = (tape[rh].value - 1) & 0xFF; break;
        case '+': tape[rh].value = (tape[rh].value + 1) & 0xFF; break;
        case '.': tape[wh].value = tape[rh].value;
                  tape[wh].token_id = tape[rh].token_id; break;
        case ',': tape[rh].value = tape[wh].value;
                  tape[rh].token_id = tape[wh].token_id; break;
        case '[':
            if (tape[rh].value == 0) {
                int depth = 1;
                ip++;
                while (ip < n && depth != 0) {
                    if (tape[ip].value == '[') depth++;
                    else if (tape[ip].value == ']') depth--;
                    ip++;
                }
                if (depth != 0) goto done;
                continue;
            }
            break;
        case ']':
            if (tape[rh].value != 0) {
                int depth = 1;
                ip--;
                while (ip >= 0 && depth != 0) {
                    if (tape[ip].value == ']') depth++;
                    else if (tape[ip].value == '[') depth--;
                    ip--;
                }
                if (depth != 0) goto done;
                ip++;
                continue;
            }
            break;
        default:
            break;
        }
        ip++;
    }
done:
    return steps;
}

/*
 * Simple xoshiro128** PRNG (per-thread, seeded from global seed).
 */
typedef struct { uint32_t s[4]; } Rng;

static inline uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

static uint32_t rng_next(Rng *r) {
    uint32_t result = rotl(r->s[1] * 5, 7) * 9;
    uint32_t t = r->s[1] << 9;
    r->s[2] ^= r->s[0]; r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2]; r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = rotl(r->s[3], 11);
    return result;
}

static void rng_seed(Rng *r, uint64_t seed) {
    r->s[0] = (uint32_t)(seed);
    r->s[1] = (uint32_t)(seed >> 16) ^ 0x9E3779B9u;
    r->s[2] = (uint32_t)(seed >> 32) ^ 0x6C62272Eu;
    r->s[3] = (uint32_t)(seed >> 48) ^ 0x61C88647u;
    for (int i = 0; i < 8; i++) rng_next(r);
}

static uint32_t rng_below(Rng *r, uint32_t n) {
    return rng_next(r) % n;
}

/*
 * Exported: run one full epoch (pop_size interactions) on the soup.
 * Sequential — use run_epochs for the multi-threaded batch variant.
 */
void run_epoch(Cell *soup, int *steps_out, int pop_size,
               int max_steps, uint64_t epoch_seed) {
    Rng rng;
    rng_seed(&rng, epoch_seed);
    Cell tape[TAPE_LEN];

    for (int k = 0; k < pop_size; k++) {
        int i = rng_below(&rng, pop_size);
        int j;
        do { j = rng_below(&rng, pop_size); } while (j == i);

        Cell *pi = &soup[i * PROG_LEN];
        Cell *pj = &soup[j * PROG_LEN];

        memcpy(tape, pi, PROG_LEN * sizeof(Cell));
        memcpy(tape + PROG_LEN, pj, PROG_LEN * sizeof(Cell));

        steps_out[k] = run_bff(tape, TAPE_LEN, max_steps);

        memcpy(pi, tape, PROG_LEN * sizeof(Cell));
        memcpy(pj, tape + PROG_LEN, PROG_LEN * sizeof(Cell));
    }
}

/*
 * Exported: run N epochs, collecting steps for each.
 *
 * When OMP_NUM_THREADS > 1, uses a persistent thread team with one spawn for
 * the entire batch.  Falls back to the fast sequential path for 1 thread to
 * avoid OpenMP overhead (~2x) that dominates at small population sizes.
 *
 * steps_out: int[n_epochs * pop_size]
 * base_seed: seeds for epochs are base_seed + epoch_index
 */
void run_epochs(Cell *soup, int *steps_out, int pop_size,
                int max_steps, int n_epochs, uint64_t base_seed) {

    if (omp_get_max_threads() <= 1) {
        for (int e = 0; e < n_epochs; e++)
            run_epoch(soup, steps_out + e * pop_size, pop_size,
                      max_steps, base_seed + e);
        return;
    }

    int *pair_i = (int *)malloc(pop_size * sizeof(int));
    int *pair_j = (int *)malloc(pop_size * sizeof(int));

    #pragma omp parallel
    {
        Cell tape[TAPE_LEN];

        for (int e = 0; e < n_epochs; e++) {
            #pragma omp single
            {
                Rng rng;
                rng_seed(&rng, base_seed + e);
                for (int k = 0; k < pop_size; k++) {
                    pair_i[k] = rng_below(&rng, pop_size);
                    do { pair_j[k] = rng_below(&rng, pop_size); } while (pair_j[k] == pair_i[k]);
                }
            }
            /* implicit barrier: all threads see the pairs before proceeding */

            #pragma omp for schedule(static)
            for (int k = 0; k < pop_size; k++) {
                Cell *pi = &soup[pair_i[k] * PROG_LEN];
                Cell *pj = &soup[pair_j[k] * PROG_LEN];

                memcpy(tape, pi, PROG_LEN * sizeof(Cell));
                memcpy(tape + PROG_LEN, pj, PROG_LEN * sizeof(Cell));

                steps_out[e * pop_size + k] = run_bff(tape, TAPE_LEN, max_steps);

                memcpy(pi, tape, PROG_LEN * sizeof(Cell));
                memcpy(pj, tape + PROG_LEN, PROG_LEN * sizeof(Cell));
            }
            /* implicit barrier: all interactions done before next epoch */
        }
    }

    free(pair_i);
    free(pair_j);
}

/*
 * Exported: initialize soup with random bytes.
 * token_ids are sequential starting from 0.
 */
void init_soup(Cell *soup, int pop_size, uint64_t seed) {
    Rng rng;
    rng_seed(&rng, seed);
    int total = pop_size * PROG_LEN;
    for (int i = 0; i < total; i++) {
        soup[i].value = rng_next(&rng) & 0xFF;
        soup[i].token_id = i;
    }
}

/*
 * Exported: count unique token_ids in the soup.
 */
int count_unique_tokens(const Cell *soup, int total_cells) {
    /* Use a hash set via open addressing */
    int cap = total_cells * 2;
    if (cap < 16) cap = 16;
    int64_t *table = (int64_t *)calloc(cap, sizeof(int64_t));
    int64_t *flags = (int64_t *)calloc(cap, sizeof(int64_t));
    int count = 0;

    for (int i = 0; i < total_cells; i++) {
        int64_t tid = soup[i].token_id;
        uint32_t h = (uint32_t)((uint64_t)tid * 2654435761ULL) % cap;
        while (1) {
            if (flags[h] == 0) {
                flags[h] = 1;
                table[h] = tid;
                count++;
                break;
            }
            if (table[h] == tid) break;
            h = (h + 1) % cap;
        }
    }
    free(table);
    free(flags);
    return count;
}

/*
 * Exported: get raw byte values for compression-based entropy.
 * out: uint8_t[total_cells]
 */
void get_values(const Cell *soup, uint8_t *out, int total_cells) {
    for (int i = 0; i < total_cells; i++) {
        out[i] = soup[i].value;
    }
}
