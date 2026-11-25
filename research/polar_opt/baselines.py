"""PyTorch reference orthogonalizers for real tensors of shape (..., rows, columns).

Polynomial arithmetic follows the cited algorithms. Performance reflects these
eager, dense PyTorch implementations.
"""

from dataclasses import dataclass
from math import prod, sqrt

import numpy as np
import torch
from numpy.polynomial import Polynomial
from torch import Tensor


# NoahAmsel/PolarExpress at f717e2c6b3ef6f67a5e00ab7f5371a85377e6fbe.
def optimal_cubic(l: float, u: float) -> tuple[float, float]:
    """Return the minimax odd cubic approximating 1 on [l, u]."""
    if not 0 <= l <= u or u <= 0:
        raise ValueError("Expected 0 <= l <= u and u > 0")
    alpha = sqrt(3 / (u**2 + l * u + l**2))
    beta = 4 / (2 + l * u * (l + u) * alpha**3)
    return 1.5 * alpha * beta, -0.5 * alpha**3 * beta


def optimal_quintic(l: float, u: float) -> tuple[float, float, float]:
    """Return the minimax odd quintic using the simplified Remez iteration."""
    if not 0 <= l <= u or u <= 0:
        raise ValueError("Expected 0 <= l <= u and u > 0")
    if 1 - 5e-6 <= l / u:
        return (15 / 8) / u, (-10 / 8) / u**3, (3 / 8) / u**5
    q, r = (3 * l + u) / 4, (l + 3 * u) / 4
    error = float("inf")
    for _ in range(100):
        system = [
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ]
        a, b, c, next_error = np.linalg.solve(system, np.ones(4))
        if abs(next_error - error) <= 1e-15:
            return float(a), float(b), float(c)
        error = next_error
        q, r = np.sqrt(
            (-3 * b + np.array([-1, 1]) * sqrt(9 * b**2 - 20 * a * c)) / (10 * c)
        )
    raise ArithmeticError("Remez iteration did not converge")


def optimal_composition(
    l: float,
    num_iters: int,
    degree: int = 5,
    safety_factor_eps: float = 0,
    cushion: float = 0,
) -> list[tuple[float, ...]]:
    """Generate the upstream sequence of odd minimax polynomials."""
    solvers = {3: optimal_cubic, 5: optimal_quintic}
    if degree not in solvers or not 0 <= l <= 1:
        raise ValueError("Expected degree 3 or 5 and 0 <= l <= 1")
    u = 1.0
    x = Polynomial.identity()
    coefficients = []
    for step in range(num_iters):
        polynomial = x * Polynomial(solvers[degree](max(l, cushion * u), u))(x**2)
        if cushion * u > l:
            polynomial *= 2 / (polynomial(l) + polynomial(u))
        if step < num_iters - 1:
            polynomial = polynomial(x / (1 + safety_factor_eps))
        coefficients.append(tuple(float(c) for c in polynomial.coef[1::2]))
        l = polynomial(l)
        u = 2 - l
    return coefficients


_JORDAN = (3.4445, -4.7750, 2.0315)
_AMSEL_COEFFICIENTS = tuple(
    optimal_composition(1e-3, 10, degree=5, safety_factor_eps=1e-2, cushion=0.02)
)

# Dao-AILab/gram-newton-schulz at e45d0aca7083cb275c9a303220c05c4abecd9187.
POLAR_EXPRESS_COEFFICIENTS = tuple(
    (a / 1.05, b / 1.05**3, c / 1.05**5)
    for a, b, c in (
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    )
)

# zhehangdu/polar-switch at 6f382f1e4583804c62a2ebaa44917995bea8111f (MIT).
# Each row computes B[dst] = a * B[src]² + b * B[add] + c * I.
PS2_55 = (
    (0, (
        (1, 0, 0, 1, -1.039372688617118, 0.0011376825939118764),
        (0, 1, 1, 164.07644513428048, 49.58061223879703, 0.000500089059050944),
        (1, 0, 0, 1, 4.238519015111072, 2.9096036081096996),
        (0, 1, 0, 1, -0.7099643179341211, -3.188845137257469),
        (1, 0, 1, 1, 0.10116427259090495, 0.36987195573416237),
    )),
    (0, (
        (1, 0, 0, 3.6023274985419302, -6.053717133712312, 0.03621204508077079),
        (0, 1, 1, 1, 2.8221769622596162, 1.4534435713646865),
        (1, 0, 1, 1, 0.5287633553256361, -0.15855532053714858),
        (0, 1, 0, 1, -0.3880284725332113, -0.745819141374713),
        (1, 0, 1, 1, 2.072314176413394, 3.287628403409071),
    )),
)

PS3_55 = (
    (0, (
        (1, 0, 0, -1, 1.0372630529657008, -0.0011080641835321228),
        (2, 1, 1, -169.7011946136095, 50.90200678616221, -0.0004999998950287138),
        (1, 2, 2, -1, 4.259459173066772, -2.9027674651455544),
        (2, 1, 2, 1, 0.6326518033906352, -3.1055845493128005),
        (0, 2, 0, 1, -0.5157563645706955, 0.5614111759107233),
    )),
    (0, (
        (1, 0, 0, 7.281110320128563, -7.974535412410232, 0.041676558977155734),
        (2, 1, 1, 1, 2.6715673993971243, 1.495553622588639),
        (0, 2, 0, 1, -1.103759381672208, -0.3426296621352286),
        (1, 0, 1, 1, -0.39530569914337854, -1.172904844980118),
        (0, 1, 0, 1, 2.6713299568702316, 3.4991981111096626),
    )),
)
_SWITCH_TABLES = {"PS2_55": (PS2_55, 2), "PS3_55": (PS3_55, 3)}


def svd_polar_fp64(G: Tensor) -> Tensor:
    """Return the reduced FP64 SVD product U @ Vh."""
    options = {"driver": "gesvd"} if G.is_cuda else {}
    U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False, **options)
    return U @ Vh


def _direct_bf16(G: Tensor, coefficients, safety: float) -> Tensor:
    tall = G.shape[-2] > G.shape[-1]
    X = G.bfloat16()
    if tall:
        X = X.mT
    norm = X.norm(dim=(-2, -1), keepdim=True)
    if safety != 1.0:
        norm = norm * safety
    X = X / (norm + 1e-7)
    for a, b, c in coefficients:
        A = X @ X.mT
        B = b * A + (c * A) @ A
        X = a * X + B @ X
        del A, B
    return X.mT if tall else X


def jordan_ns5(G: Tensor, steps: int = 5) -> Tensor:
    """Keller Jordan's BF16 Muon quintic with denominator ||G_bf16||_F + 1e-7."""
    return _direct_bf16(G, _take_steps((_JORDAN,), steps), safety=1.0)


def _take_steps(coefficients, steps: int):
    if steps < 0:
        raise ValueError("Steps must be nonnegative")
    return coefficients[:steps] + (coefficients[-1],) * max(0, steps - len(coefficients))


def polar_express(G: Tensor, steps: int = 5) -> Tensor:
    """Amsel et al.'s BF16 Polar Express with denominator 1.01*||G_bf16||_F + 1e-7."""
    return _direct_bf16(G, _take_steps(_AMSEL_COEFFICIENTS, steps), safety=1.01)


def _normalized_fp16(G: Tensor) -> Tensor:
    X = G.reshape(-1, *G.shape[-2:]).float()
    return (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).half().contiguous()


def _gram_run(X: Tensor, coefficients, resets: set[int], tall: bool) -> Tensor:
    R = X.mT @ X if tall else X @ X.mT
    identity = torch.eye(R.shape[-1], dtype=X.dtype, device=X.device)
    for i, (a, b, c) in enumerate(coefficients):
        if i in resets:
            X = X @ Q if tall else Q @ X
            R = X.mT @ X if tall else X @ X.mT
            del Q
        Z = torch.baddbmm(R, R, R, alpha=c, beta=b)
        if i == 0 or i in resets:
            Q = Z + a * identity
        else:
            Q = torch.baddbmm(Q, Q, Z, beta=a)
        if i + 1 < len(coefficients) and i + 1 not in resets:
            RZ = torch.baddbmm(R, R, Z, beta=a)
            R = torch.baddbmm(RZ, Z, RZ, beta=a)
            del RZ
        del Z
    return X @ Q if tall else Q @ X


def gram_ns(G: Tensor, steps: int = 5, reset_iterations=(2,)) -> Tensor:
    """Dao's FP16 Gram recurrence, including direct quintics for square inputs.

    Normalize in FP32 with denominator ||G_fp32||_F + 1e-7. Restart before
    zero-based iteration 2 by default, and restore the original dtype.
    """
    X = _normalized_fp16(G)
    coefficients = _take_steps(POLAR_EXPRESS_COEFFICIENTS, steps)
    rows, columns = X.shape[-2:]
    if rows == columns:
        for a, b, c in coefficients:
            A = X @ X.mT
            B = torch.baddbmm(A, A, A, alpha=c, beta=b)
            X = torch.baddbmm(X, B, X, beta=a)
            del A, B
    elif coefficients:
        X = _gram_run(X, coefficients, set(reset_iterations) - {0}, tall=rows > columns)
    return X.to(G.dtype).reshape(G.shape)


def polar_switch(G: Tensor, variant: str = "PS2_55") -> Tensor:
    """Run Du and Su's FP16 circuit with FP32 epilogues and the input dtype on return."""
    variant = variant.upper().replace("-", "_")
    if variant not in _SWITCH_TABLES:
        raise ValueError(f"Unsupported Polar Switch variant {variant!r}")
    table, buffers = _SWITCH_TABLES[variant]
    X = _normalized_fp16(G)
    tall = X.shape[-2] > X.shape[-1]
    n = min(X.shape[-2:])
    work = [X.new_empty(X.shape[0], n, n) for _ in range(buffers)]
    for residual, iterations in table:
        view = X.mT if tall else X
        torch.bmm(view, view.mT, out=work[0])
        for dst, src, add, a, b, c in iterations:
            A = work[src]
            if X.is_cuda:
                # FP16 tensor-core operands, with FP32 output through the diagonal shift.
                value = torch.baddbmm(work[add], A, A.mT, alpha=a, beta=b,
                                      out_dtype=torch.float32)
            else:
                A = A.float()
                value = torch.baddbmm(work[add].float(), A, A.mT, alpha=a, beta=b)
            value.diagonal(dim1=-2, dim2=-1).add_(c)
            work[dst].copy_(value)
            del value
        B = work[dst]
        left, right = (X, B) if tall else (B, X)
        X = torch.baddbmm(X, left, right, beta=residual) if residual else left @ right
    return X.to(G.dtype).reshape(G.shape)


@dataclass(frozen=True)
class TheoreticalCost:
    """Leading matrix-product FLOPs and live matrix storage for one invocation."""

    flops: int
    flops_formula: str
    symmetric_flops: int | None
    symmetric_flops_formula: str | None
    peak_intermediate_bytes: int


def theoretical_cost(
    method: str, shape: tuple[int, ...], *, steps: int = 5,
    reset_iterations=(2,), input_dtype: torch.dtype = torch.float32, device="cuda",
) -> TheoreticalCost:
    """Estimate live matrix bytes excluding caller input and backend workspace.

    Count the output during its construction. Dense FLOPs describe this backend.
    Symmetric FLOPs assume a kernel computes only one triangle when applicable.
    Both exclude lower-order elementwise work. SVD uses a conventional
    4*m*n² + 8*n³ thin-SVD model, plus 2*m*n² for U @ Vh.
    """
    rows, columns = shape[-2:]
    n, m = sorted((rows, columns))
    rectangular, square = n * m, n * n
    batches = prod(shape[:-2])
    element = torch.empty((), dtype=input_dtype).element_size()
    if method == "svd_polar_fp64":
        return TheoreticalCost(
            batches * (6 * m * n**2 + 8 * n**3), "(6α + 8)n³ (SVD estimate)",
            None, None, batches * 8 * (2 * rectangular + square + n),
        )
    if method in ("jordan_ns5", "polar_express"):
        dense_coeffs = (4 * steps, 2 * steps)
        symmetric_coeffs = (3 * steps, steps)
        # X, A, B and output/pointwise temporaries can be simultaneously live.
        peak = 2 * (4 * rectangular + 2 * square)
    elif method == "gram_ns" and n == m:
        dense_coeffs = (4 * steps, 2 * steps)
        symmetric_coeffs = (3 * steps, steps)
        peak = max(8 * rectangular, 2 * (2 * rectangular + 2 * square),
                   (2 + element) * rectangular)
    elif method == "gram_ns":
        segments = (1 + len({i for i in reset_iterations if 0 < i < steps})) if steps else 0
        dense_coeffs = (4 * segments, 2 * (4 * steps - 3 * segments))
        symmetric_coeffs = (3 * segments, 4 * steps - 3 * segments)
        # The caller holds the initial X while the recurrence can replace it at restarts.
        held = rectangular if segments > 1 else 0
        peak = max(8 * rectangular, 2 * (held + rectangular + 6 * square),
                   2 * (held + 2 * rectangular + 3 * square), (2 + element) * rectangular)
    elif method in _SWITCH_TABLES:
        _, buffers = _SWITCH_TABLES[method]
        dense_coeffs = (8, 20)
        symmetric_coeffs = (6, 10)
        # PyTorch materializes an FP32 square for the diagonal shift.
        square_scratch = (4 if torch.device(device).type == "cuda" else 12) * square
        peak = max(8 * rectangular, 4 * rectangular + 2 * buffers * square,
                   2 * rectangular + 2 * buffers * square + square_scratch,
                   (2 + element) * rectangular + 2 * buffers * square)
    else:
        raise ValueError(f"Unknown baseline {method!r}")

    def flops(coefficients):
        wide, narrow = coefficients
        return batches * (wide * m * n**2 + narrow * n**3), f"({wide}α + {narrow})n³"

    dense, dense_formula = flops(dense_coeffs)
    symmetric, symmetric_formula = flops(symmetric_coeffs)
    return TheoreticalCost(dense, dense_formula, symmetric, symmetric_formula, batches * peak)
