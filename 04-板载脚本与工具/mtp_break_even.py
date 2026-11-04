"""MTP speculative-decoding break-even calculator for the AX7020 Qwen3.5-0.8B engine.

Pure python3, no board dependency.  Takes the numbers measured on the board and
prints the break-even acceptance rate alpha* and the speedup at several alphas.

Measured inputs
  T1   = 930 ms   one forward pass with the engine's own C/NEON kernels
                   (605 ms GEMV + 325 ms everything else), pl_run.py --cpu
  GEMV = 605 ms   the 150 BFP8 tensors, 497 MB of mantissas per token
  OTH  = 325 ms   lm_head 117 + gdn 74 + conv 17 + swiglu 12 + Python ~105

  Batch GEMV scaling, gemv_batch_bench on all 7 real shapes (OMP_NUM_THREADS=2),
  aggregated over the 150-tensor mix:
      B=1 1535.4 ms   B=2 2634.6   B=3 3766.2   B=4 4961.4
  -> T(B)/T(1) = 1.716 / 2.453 / 3.231, i.e. every extra verified position costs
     ~72-74% of a full pass.  The kernel is compute-bound, not bandwidth-bound.

Two bounds on the non-GEMV part:
  A "naive batch"   : only the GEMV kernel batches, everything else runs K times
                      -> OTH_K = K * 325      (worst case, upper bound on alpha*)
  B "ideal batch"   : everything else amortises too
                      -> OTH_K = 325          (best case, lower bound on alpha*)
The real engine can do neither today: there is no batched forward pass at all.

Usage:  python3 mtp_break_even.py
"""

T1 = 930.0
GEMV = 605.0
OTH = 325.0
RATIO = {2: 1.7159, 3: 2.4529, 4: 3.2313}
BASE_NEW_TOK_S = 0.717        # measured end-to-end, generate(..., 8)
BASE_DECODE_TOK_S = 1.075     # measured per forward pass


def t_k(k, amortise):
    g = GEMV * RATIO[k]
    return g + (OTH if amortise else OTH * k)


def exact_tokens(alpha, k):
    """Expected accepted tokens per verification round: 1 + a + a^2 + ... + a^K."""
    return sum(alpha ** i for i in range(k + 1))


def alpha_star(k, amortise):
    """Solve exact_tokens(alpha, k) == T_K / T1 by bisection."""
    target = t_k(k, amortise) / T1
    lo, hi = 0.0, 0.999
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if exact_tokens(mid, k) < target:
            lo = mid
        else:
            hi = mid
    return lo


def alpha_star_lead(k, amortise):
    """The lead's formula: (1 + alpha*K) / T_K = 1 / T1  ->  alpha* = (T_K/T1 - 1)/K."""
    return (t_k(k, amortise) / T1 - 1.0) / k


print('measured: T1=%.0f ms (GEMV %.0f + other %.0f), 497 MB of weights/token'
      % (T1, GEMV, OTH))
print('')
print('verification cost T_K and break-even alpha*')
print('K  T_K worst  a* worst  a* worst  |  T_K best  a* best  a* best')
print('   (ms)      (lead)    (exact)   |   (ms)     (lead)   (exact)')
for k in (2, 3, 4):
    tw, tb = t_k(k, False), t_k(k, True)
    print('%d  %9.0f  %8.3f  %8.3f  |  %8.0f  %7.3f  %7.3f'
          % (k, tw, alpha_star_lead(k, False), alpha_star(k, False),
             tb, alpha_star_lead(k, True), alpha_star(k, True)))

print('')
print('speedup vs T1, exact expected accepted length (lead formula in brackets)')
print('K  alpha  worst-case   best-case    new tok/s worst/best  (base %.3f)'
      % BASE_NEW_TOK_S)
for k in (2, 3, 4):
    for a in (0.3, 0.5, 0.75):
        e = exact_tokens(a, k)
        e_lead = 1.0 + a * k
        sw = e * T1 / t_k(k, False)
        sb = e * T1 / t_k(k, True)
        sw_l = e_lead * T1 / t_k(k, False)
        sb_l = e_lead * T1 / t_k(k, True)
        print('%d  %.2f   %.3fx (%.3fx)  %.3fx (%.3fx)   %.3f / %.3f'
              % (k, a, sw, sw_l, sb, sb_l,
                 BASE_NEW_TOK_S * sw, BASE_NEW_TOK_S * sb))

print('')
print('best achievable new-token rate at alpha=0.75: %.3f tok/s'
      % (BASE_NEW_TOK_S * max(exact_tokens(0.75, k) * T1 / t_k(k, True)
                              for k in (2, 3, 4))))
print('PS read wall (measured 1259 MB/s): 741.6 MB/token -> %.0f ms -> %.2f tok/s'
      % (741.6 / 1.259, 1000.0 / (741.6 / 1.259)))
