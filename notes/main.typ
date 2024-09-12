#import "@preview/charged-ieee:0.1.2": ieee
#import "@preview/ctheorems:1.1.2": *
#show: thmrules.with(qed-symbol: $square$)

// Latex look (https://typst.app/docs/guides/guide-for-latex-users/#latex-look)
#set page(margin: 1.75in)
#set par(leading: 0.55em, first-line-indent: 1.8em, justify: true)
#set text(font: "New Computer Modern")
#show raw: set text(font: "New Computer Modern Mono")
#show par: set block(spacing: 0.55em)
#show heading: set block(above: 1.4em, below: 1em)

// Theorem environments
#let theorem = thmbox("theorem", "Theorem", inset: (x: 0em, y: 0em), base: none)
#let corollary = thmplain(
	"corollary", "Corollary", base: "theorem", 
	inset: (x: 0.0em, top: 0em), titlefmt: strong
)
#let lemma = thmbox("lemma", "Lemma", inset: (x: 0em, y: 0em), base: none)
#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em))
#let example = thmplain("example", "Example", inset: (x: 0em, top: 0.5em)).with(numbering: none)
#let proof = thmproof("proof", "Proof")

// Attach appendix
#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  counter(heading).update(0)
  body
}

// Show page counts 
#set page(footer: context [
  *Topological Deep learning with the Heat Kernel*
  #h(1fr)
  #counter(page).display(
    "1/1",
    both: true,
  )
])

// Metadata
#show: ieee.with(
	title: [Topological Deep Learning with the Heat Kernel],
	abstract: [
		Relational structures, such as graphs and hypergraphs, provide a versatile framework for modeling entity interactions across multiple domains. 
		With the rise of *geometric deep learning* models, such graph convolutional networks (GCNs), these structures have become central to modern machine learning pipelines. 
		Inspired by their success in pose-invariant shape analysis applications, we introduce a novel framework for encoding higher-order (hyperedge) interactions based on simplicial diffusion. 
		Central to our approach is the matrix functional calculus, which we exploit to develop a highly scalable algorithm for extracting provably informative and multsicale features from hypergraphs.
		Confirming recent work on the topic, our empirical analysis suggests that from a learning perspective, a weighted simplicial complex conveys as much information as a hypergraph does without loss of information. 
	],
	authors: (
		(
			name: "Matt Piekenbrock",
			department: [Khoury College of Computer Sciences],
			organization: [Northeastern University],
			location: [Boston, Massachusetts],
			email: "piekenbrock.m@northeastern.edu"
		), 
		(
			name: "Jose Perea",
			department: [
				Khoury College of Computer Sciences, \ College of Mathematics
			],
			organization: [Northeastern University],
			location: [Boston, Massachusetts],
			email: "j.pereabenitez @ northeastern.edu"
		)
	),
	index-terms: ("Geometric Deep Learning", "Graph Neural Networks", "Topological data analysis", "Linear Algebra"),
	bibliography: none,
)


= Introduction <sec_introduction>

Many real-world datasets involve complex, higher-order relationships that cannot be fully characterized by pairwise interactions alone. Hypergraphs are a natural and powerful way to represent such higher order structures; however, their lack of structure presents challenges for learning tasks, preventing the full utilization of techniques from *geometric deep learning* @bronstein2017geometric, which aims to generalize deep neural network models to non-Euclidean spaces (e.g graphs, manifolds). Indeed, recent work has shown certain simplicial relaxations of higher-order data can efficiently surpass existing graph-based learning tasks @yang2022efficient. 
// Simplicial complexes, being more structured and able to capture higher-order relationships in a hierarchical manner, provide a natural extension of these techniques, enabling more expressive learning models.

Building on these ideas, we propose a simple and efficient framework to extend the scope of hypergraphs to the more structured simplicial domain. Following the seminal work of @baccini2022weighted, our method encodes hypergraphs as weighted simplicial complexes without loss of information. One notable application of this conversion is the exploitation of the heat kernel's properties, including the generation of multiscale, informative signatures that effectively capture higher-order interactions.

== Overview <sec_overview>

We briefly summarize our approach. Let $H = (V, cal(E))$ denote undirected hypergraph with which one wishes to extract features for learning purposes. Our proposed framework for producing such as featurization is as follows: 

1. Construct the $d$-dimensional simplicial closure $S$ of $H$:

$ S(H) = { sigma subset.eq e : e in cal(E) } $

2. Define a map $w: S -> bb(R)_+$ comprised of each simplex's *topological weight* $w_sigma$ and its *affinity weight* $omega_sigma$:
	
	$ w(sigma) = omega_sigma + sum_(sigma' in op("cofacet")(sigma)) w_sigma' $ <eq_weight_def>
	
	We discuss ways to compute these weights in @sec_weight_assignment.

3. To capture $H$'s higher-order interactions, construct a $p$-Laplacian $cal(L)_p$ from $S$, such as the _Hodge laplacian_: 

$ cal(L)_p = L_p^(op("up")) (w) + L_p^(op("dn")) (w) $

4. The final featurization $X$ on the $p$-simplices is defined by the diagonal of the a chosen matrix function: 
	$ X := op("diag")(f(cal(L)_p)) = op("diag")(U f(Lambda) U^T ) $ <eq_diag_mf>
	where $f : bb(R)^n -> bb(R)$ is an appropriate spectral function.

Our proposed featurization method @eq_diag_mf is succinct in that it is $O(n)$-sized for each function $f$, where $n = abs(S_p)$ is the number $p$-simplices in $S$. In @sec_diffusion we show that, for certain types of functions $f$ (e.g. those deriving from the matrix exponential), the corresponding feature vectors are provably _informative_, _stable_, and _multiscale_. 

// We leave the rest of the paper to discuss how to parameterize the weight map $w$ and spectral function $f$. 


= Background work <sec_background>

TODO 
// Both the heat kernel and diffusion processes have long been used in both the context of graphs and for learning purposes. Indeed, a graph convolution 

// The heat kernel can serve as a specific graph convolution kernel when using the exponential decay of heat diffusion as a filter.

// Others have tried to quantify the degree to simplicial complexes and hypergraph differ in their representation power, such as in @landry2024simpliciality.
// In contrast, the properties of simplicial complexes, their asymptotic behavior, and their connections to random walks and diffusion processes via combinatorial Laplacians are well established, motivating the use of techniques from algebraic topology and spectral graph theory. 

= Notation <sec_notation>

In this work, a _hypergraph_ (or family of sets) is a pair $H = (V, cal(E))$ where $V$ is a finite set of elements called _vertices_ and $cal(E) subset.eq cal(P)(V)$ is any subset of the power set $cal(P)(V)$ of $V$. A _simplicial complex_ is a pair $S = (V, Sigma)$ satisfying: 
1. if $v in V$, then ${v} in Sigma$ 
2. for any $sigma in Sigma$, $tau subset sigma => tau in Sigma$. 
Denote by $S_p$ the set of $p$-simplices of $S$. An _oriented_ simplex $[sigma]$ is a simplex $sigma in S$ with an ordering on its vertices. For simplicity, we will always assume to be induced by an ordering on $V$. A $p$-chain is a formal $bb{R}$-linear combination of $p$-simplices of $S$; The collection of $p$-chains under addition yields an vector space denoted $C_p(S, bb(R))$ with basis ${[sigma] : sigma in S_p }$. 

We will often characterize simplicial complexes via their oriented boundary operators. The boundary $partial_p [sigma]$ of an oriented $p$-simplex $[sigma] in S$ is defined as the alternating sum of its oriented co-dimension 1 faces, which collectively for all $sigma in S^p$ define the $p$-th _boundary matrix_ $partial_p : C(p) -> C(p-1)$:
$ 
partial_p [i,j] = cases(
	(-1)^(s_(i j)) & sigma_i in partial[sigma_j], 
		0 & text("otherwise")
	)
$
where $s_(i j) = op("sgn")([sigma_i], partial [sigma_j])$ records the orientation. 
Denote by $partial_p^ast : C_(p-1) -> C_p$ the adjoint of $partial_p$. The $p$-th _combinatorial Laplacian_ $cal(L)_p : C_p (S, bb(R)) -> C_p (S, bb(R))$ is defined as: 

$ Delta_p := partial_(p+1) circle.stroked.small med partial_(p+1)^ast + partial_p^ast med circle.stroked.small med partial_p $

For an overview on the spectra of combinatorial Laplacians, see @horak2013spectra. 

// A _weight function_ of $S$ is any positive function $w: S -> (0, infinity)$, i.e. any function that maps every simplex $sigma in S$ to a strictly positive weight. $S$ is called _unweighted_ is $w(sigma) = 1$ for all $sigma in S$.


= Methodology <sec_methodology>

There are many ways to map higher-order interactions, such as hyperedges $cal(E)$, to weighted simplices. Given a hypergraph $H$ and its simplicial closure $S$, one such lossless encoding is given by the trivial identity map: 

// $ w(sigma) = cases(med 1 wide& sigma in H, med 0 wide& sigma in.not H ) $
$ w&: med& S & -> bb(R)_+ \ 
	 &	med& sigma & |-> bb(1)(sigma in cal(E)) $ <eq_trivial_map>

By construction, the closure $S$ must contain a face $sigma in S$ for every hyperedge $e in cal(E)$, thus the map is well-defined. 

The main issue with the map from @eq_trivial_map—aside from the potential size of $S$—is that its dynamics are less understood due to its lack of well-defined asymptotic behavior. 
Indeed, given any $p in bb(N)$, any choice of weight function defines an inner product $angle.l [sigma], [sigma']angle.r_w$ on the chain group $C_p (S, bb(R))$ as follows:  
$ angle.l [sigma], [sigma']angle.r_w := delta_(sigma sigma') dot (w(sigma))^(-1), med forall sigma, sigma' in S_p $
Similarly, $angle.l dot, dot angle.r_w$ induces an inner product on the cochain space $C^p (S, bb(R))$ of $S$: 
$ norm( angle.l f comma g angle.r )_w = sum_(sigma in S_p) w(sigma) dot f([sigma]) g([sigma])) $ 
There is a one-to-one correspondence between weight functions and possible inner products on cochain groups $C^p(S, bb(R))$, where elementary cochains are orthogonal @horak2013spectra. 
The matrix representations of a choice of inner product is given by: 
$ cal(L)_p = underbrace(partial_(p+1) W_(p+1) partial_(p+1)^T W_p^(-1), L_p^op("up")) + underbrace(W_p partial_p^T W_(p -1)^(-1) partial_p, L_p^op("dn")) $ <eq_weighted_hodge_laplacian>
where $W_p$ denotes the diagonal representation of $w$ applied to the $p$-simplices of $S$. Clearly, @eq_weighted_hodge_laplacian is only well-defined if the corresponding weight function $w$ takes strictly positive values, preventing the use of characteristic functions akin to @eq_trivial_map. 
// Note that the topological weight depends only the dimensions of the faces of a given simplex, and thus is defined for any simplicial complex $S$.

== Choosing the weights <sec_weight_assignment> 

// Rather than working in the hypergraph domain directly, @baccini2022weighted proposed to first translate the problem to the _weighted_ simplicial domain; rather than representing hyperedges  directly, the idea is to encode the interaction information via diffusion (we will discuss this more in @sec_diffusion).
In the absence of any 'natural' map from hyperedges to weighted simplices, 
Baccini et al. @baccini2022weighted proposed to weight each simplex $sigma in S$ in the closure of $H$ via a combiniation of a _topological_ and _affinity_ weight. Specifically, for each hyperedge $e in cal(E)$ of dimension $d$, a $d$-dimensional simplex $sigma$ is constructed with each $k$-dimensional face $tau subset.eq sigma$ is assigned the following topological weight:

$ w_sigma (tau) = (d-k)! med slash med d! $ <eq_top_weight>

Thus, both singletons and pairwise interactions (edges) are assigned unit weight, whereas higher-order simplices have successively lower weights. Some of the properties of this weight function are _strict positivity_, _monotonicity with respect to the face relations_, and _normalized scale_: 

#set math.equation(numbering: none)

$ cases(
	med w(sigma) > 0 wide& text("for every face") sigma in S, 
	med w(sigma) gt.eq sum_(tau supset sigma) w(tau) wide& text("for every codim-1 ") tau in S, 
	med w(v) = sum_(e in cal(E)) bb(1)(v in cal(E)) wide& text("for all ") v in V
) $ <eq_top_weight_props>

#align(center)[
	#figure(
		image("images/bridge_net_weight.png", width: 75%),
		caption: [
			Each simplexes _topological weight_ sums of the weights of its codim-1 faces; each vertex weight matches the number of hyperedges it participates in. 
		],
	)<fig_hg_weights>
]

Moreover, this weight function is _additive_ in the sense that the weight of any simplex $sigma in S$ in the simplicial closure of $S$ of $H$ is given by summing the weights disjoint union of the hyperedges: 

$ w(sigma) = sum_(e in cal(E)) w_e (sigma) $ 

It may be readily verified that any choice of affinity weights $omega_sigma$ may be readily reconstructed from the final weight map $w : S -> bb(R)_+$ using @eq_weight_def. To see this with an example, see @fig_hg_weights.

// To sufficiently capture higher-order interactions using only simplicial structure, we use the strategy from @baccini2022weighted, which deduces the affinity weight from the number of times $c$ a face $sigma in S$ appears in a hyperedge $h in H$ of order $n$. 

// Insert more about the losslessness of the affinity and topological weights

== Choosing the featurization <sec_featurization>

To featurize the weighted simplicial complexes for learning purposes, we exploit the functional calculus of matrices, which parameterizes linear operators via weighted projections onto their eigenspaces:

$ f(A) colon.eq sum_(i=1)^n f(lambda_i) P_i $

where $P_i P_j = 0 text("for") i eq.not j$ and $sum_(i=1)^n P_i = I$. Note each projector $P_i$ is uniquely determined by the eigenspace $U_lambda_i$ generated by the pair $(lambda_i, u_i)$, i.e. the subspace of $bb(R)^n$ wherein the action of $A$ mimics scalar multiplication: 
$ U_lambda = { med u in U : A u = lambda u med } $
// $op("ker")(A - lambda_i I)$.
In the literature, these linear maps are referred to as _Lowner_ operators. One benefit of using the matrix functional calculus is that derivatives are available in closed-form. 

#theorem("Differentiability")[
	if $f : bb(R) -> bb(R)$ is continuously differentiable, then the differential $nabla F$ of $F$ is available in closed-form:
  $ nabla F(A)[H]= U ( f^[1](Lambda) med circle.stroked.small med (U^T H U)) U^T $
	where $f^[1]$ denotes the first divided differences matrix: 
	$ (f^[1](x))_(i j) = cases(f'(x_i), (f(x_i) - f(x_j)) / (x_i - x_j)) $
]

The choice of matrix and matrix function inevitably produces different featurizations, some of which have proven useful for learning purposes. Suppose $X$ represents a design matrix, $A$ an adjacency matrix a graph, and $M$ a positive semi-definite matrix. Some  exemplary matrix functions are given below: 

#set align(left)
#set list(tight: true, marker: none)
#grid(
  columns: (5fr, 0.5fr, 6fr),
  rows: (6pt, 6pt),
  gutter: 8pt,
  $f(X) = bb(1)(lambda gt.eq lambda_k)$, $<->$, "Principle Component Analysis", 
  $f(X) = exp(-t M)$, $<->$, "Matrix exponential", 
  $f(X) = (X^T X + lambda I)^(-1)$, $<->$, "Tikhonov Regularization", 
  $f(A) = (I - alpha A)^(-1)$, $<->$, "PageRank"
)
// - $f(X) = bb(1)(lambda gt.eq lambda_k)$ $thick <-> thick$ Principle Component Analysis
// - $f(M) = exp(-t M)$ $thick <-> thick$ Matrix exponential
// - $f(X) = (X^T X + lambda I)^(-1)$ $thick <-> thick$ Tikhonov Regularization


#set align(left)

For more examples of common uses of these operators, see @musco2018stability. 

== Diffusion-based signatures <sec_diffusion> 

Let $G = (V,E)$ denote a graph over $n = abs(V)$ vertices and $m = abs(E)$ edges. A _diffusion process_ across $G$ is modeled via a time-varying vector $v(t) in bb(R)^n$ where each component $v_i(t) in bb(R)$ represents the value of the diffusion at vertex $v_i$. 
If $E$ comes equipped with edge weights $w_(i j)$ representing _conductivity_, the _flow_ from $v_i$ to $v_j$ for any pair $(v_i, v_j) in V times V$ is defined by the quantity $w_(i j) (v_i (t) - v_j (t))$. Summing flows across neighbors yields the change in diffusion values: 

$ v_j'(t) = sum_(i tilde.op j) w_(i j)(v_i (t) - v_j (t)) $

This change in diffusion is captured by the _graph Laplacian_ $L$:
$ v'(t) = -L v(t) quad <-> quad L dot u(x,t) = - frac(partial u(x, t), partial t) $
With initial conditions $v(0) in bb(R)^n$ representing the amount of heart at each vertex at time $t = 0$, the solution to the partial differential equation above is given by the _Laplacian exponential diffusion kernel_: 

$ v(t) = exp(-t L) v(0) $ <eq_diffusion_process>

Where the matrix operator $exp(-t L)$, given in closed-form, is called the _heat kernel_ at time $t gt.eq 0$: 

$ exp(-t L) = H_t = U exp(-t Lambda) U' = sum_(i=1)^n e^(-t lambda_i) u_i u_i^T $

In other words, the value $v_i (t)$ describes the heat at vertex $v_i$ at time $t$ after a diffusion of heat given by $v(0)$ @lafferty2005diffusion. 

Due to its myriad of attractive properties and its intrinsic connection to diffusion,  the heat kernel is a natural choice of matrix function for characterizing higher-order interactions. Indeed, it is an _isometric invariant_ and _informative_, as shown by the following theorem.  
#theorem[Intrinsic & Informative @sun2009concise][
	Let $M$, $N$ be compact Riemannian manifolds, and let $T : M -> N$ be a surjective map. If $H_t (T(x), T(y)) = H_t (x, y)$ for any $x, y in M$ and $t > 0$, then $T$ is an isometry. Moreover, if $T$ is an isometry, then $H_t (x,y) = H_t (T(x), T(y))$ for any $x, y in M$ and $t > 0$.
]
In other words, from a shape / geometric perspective, the heat kernel contains all of the information about the intrinsic geometry of the underlying shape and hence fully characterizes shapes up to isometry.
// The heat kernel is also _multi-scale_, and _stable_ @lafferty2005diffusion.

Our interest in the heat kernel stems from the fact that it is possible to extract _stable_ features which capture higher-order interactions from it. One such featurization---the _Heat Kernel Signature_ @sun2009concise (HKS)---is often used to distinguish shapes in pose-invariant 3D contexts due to its multiscale nature. Given a Laplace-Beltrami operator $L$ with eigenpairs $(lambda_i, phi.alt_i)_(i=1)^n$, it is defined as: 

$ op("hks")_t (x) := sum_(i=1)^n e^(-t lambda_i ) phi.alt_i (x) phi.alt_j (x) $ <eq_hks>

In other words, the HKS at a fixed point $x$ is simply a restriction of the heat kernel to the temporal domain. Surprisingly, it can be shown that under relatively mild assumptions, the HKS characterizes all of the information of the Heat Kernel (see @sun2009concise). Moreover, for a fixed simplicial complex $S = (V, Sigma)$ and its corresponding Laplacian $L_p = U Lambda U^T$, it is straightforward to see from the definition above that it is in fact equivalent to the diagonal of the heat kernel.

$ op("hks"_t)(V) = op("diag")(H_t) = op("diag")(U exp(-t Lambda) U^T ) $

The stability and multiscale nature of the HKS originally motivated its use in multi-scale shape matching. As the diagonal of a particular type of matrix function, it fits squarely into our proposed framework via @eq_diag_mf. 
// To demonstrate this, we continue the example from @fig_hg_weights in @fig_diffusion_sig.
// @eq_diffusion_process
#example[
 Consider the set of all hypergraphs whose simplicial closures are given by the maximal simplices $S = {(0,1,2), (1,2,3) }$. As there are $9$ non-maximal faces in the closure, there are $2^9$ distinct hypergraphs with identical closures; for any conversion to be useful, we would like a lift capable of _distinguishing_ between hypergraphs in this set.

 Instead, in @fig_diffusion_sig we plot 5 hypergraphs and their weighted closures using @eq_weight_def. In the third column, we show a diffusion process starting with a unit amount of heat at the red vertex. Thoughout the diffusion, the red vertex (0) starts with a unit amount of heat, which it diffuses through the weighted edges (0,1) and (0,2). In the 3rd row, observe the blue vertex heats up faster than the green due to having higher conductivity, and the orange vertex (3) heats up the slowest due to being not adjacent to the heat source (0).  

 In the fourth column, we plot the HKS featurization of the bridge network at the same time points. Note this does not depend on an initial distribution of heat. Observe the HKS not only distinguishs distinct hypergraphs, but also symmetrically related graphs are handled naturally.
]

#figure(
	image("images/bridge_net_diffusion.png", width: 100%), 
	caption: [
		Hypergraphs, their weighted closures, diffusion curves, and signatures. The 'width' of each edge is inversely proportional to its weight. Note the weight function influences both the flow rate and the corresponding signatures. 
	]
) <fig_diffusion_sig>

// To demonstrate this, below is a plot of the MDS embedding computed from the Euclidean distance matrix over the HKS-features, for a heuristic choice of time points $t_1, t_2, dots, t_k$.
// It was shown that multiscale invariants such as those deriving from the heat kernel were shown in [1] to yield more *information gain* than in the unweighted settings.
// Though not exhibiting perfect symmetry, many of the distances are quite intuitive, and each of the $2^9$ distinct hypergraphs are indeed distinguished by the HKS. 
// #figure(
// 	image("images/bridge_net_mds.png", width: 80%),
// 	caption: [
// 		Hypergraph example using classical multidimensional scaling. 
// 	],
// ) <fig_diffusion_mds>

= Computation  

Paramount to the interoperability of our proposed framework with modern machine learning methods is its scalability. While the computation of @eq_diag_mf can be accomplished by standard eigen-decomposition algorithms (e.g. Cuppens divide-and-conquer), most modern featurization methods beyond quadratic complexity are considered intractable for large data sets. Yet another desirably quality for any feature extraction method is scale with the sparsity of the input. 

Fortunately, as we show below, any featurization of the form $op("diag")(f(cal(L)))$ where $cal(L)$ is a Laplacian matrix can not only be computed exactly in $O(n^2)$ time and $O(n)$ space, but can also be iteratively $(1 plus.minus epsilon)$-approximated. 

== Exact computation 

The heart of both the exact and approximate computations is the _Lanczos method_. Given a symmetric $A in bb(R)^(n times n)$ with eigenvalues $lambda_1 gt.eq lambda_2 > dots gt.eq lambda_r > 0$ and a vector $v eq.not 0$, the Lanczos method generates the triple $(K, Q, T)$:
$ 
K &= [med A^0 v bar.v A^1 v bar.v A^2 v bar.v dots bar.v A^(r-1)v med ]  \
Q &= [med q_1 bar.v q_2 bar.v dots bar.v q_r med] arrow.l.long op("qr")(K) \
T &= Q^T A Q 
$
where $Q$ is orthogonal, $K$ is called the _Krylov matrix_ with respect to $(A, v)$, and $T$ has a tridiagonal form. It can be shown that $T$ is similar to $A$, and that its eigenvalues $Lambda (T)$ may be obtained in $O(n^2)$ time (and $O(n)$ space), reducing the problem to obtaining $T$ itself. 

A classical result, due to Lanczos' _three-term recurrence_, shows that none of these matrices $(K, Q, T)$ need be formed explicitly@paige1972computational; at most three vectors $(q_(i-1), q_i, q_(i+1))$ need to be stored in memory at any given time. This is perhaps best demonstrated by the following result: 
// This is best summarized by Paige et al@paige1972computational, though see also: 

#lemma[Lanczos complexity @simon1984analysis][
Given a symmetric rank-$r$ matrix $A in bb(R)^(n times n)$ whose matrix-vector operator $A |-> A x$ requires $O(eta)$ time and $O(nu)$ space, the Lanczos iteration computes $Lambda(A) = { lambda_1, lambda_2, dots, lambda_r }$ in $O(max{eta, n} dot r)$ time and $O(max{nu, n})$ space, when executed in exact arithmetic. 
] <lemma_lanczos_quadratic>

Clearly, @lemma_lanczos_quadratic implies a $O(n r)$ time and $O(n)$ space computation route so long as we can guarantee the action $v |-> A v$ is efficient enough and we allow for exact arithmetic. To address the former, we prove the following result: 

#lemma[Laplacian matvec][
	For any constant $p > 0$, the action $v |-> cal(L)_p v$ of any combinatorial $p$-Laplacian $cal(L)_p$ constructed from a simplicial complex $S$ can be computed at most $O(m)$ time and $O(n)$ space, where $m = abs(S_(p+1))$ and $n = abs(S_p)$, respectively. 
]
// #corollary[Linear space complexity][
// 	Any featurization that can be written of the form given by @eq_diag_mf, i.e. as the diagonal of a a matrix function of a combinatorial Laplacian operator, can be computed using $O(max(n, m))$ space, where $n = abs(S_p)$ and $m = abs(S_(p+1))$ are the number of $p$- and $(p+1)$-simplices in $S$, respectively. 
// ] <cor_space_complexity> 

To address the latter, ideally we would like to extend the analysis to a more realistic computation model (e.g. finite-precision WordRAM). However, the Lanczos method is known to break down from numerical issues that not only muddle the convergence behavior, but also pose barriers to analysis. Indeed, it's well known in practice that accurate eigenvalue computation on the interior of the spectrum is infeasible to do with Lanczos in finite-precision unless post-processing method are added to the computation, such as partial reorthogonalization (cite) or deflation techniques (cite). Unfortunately, such techniques often increase computation time substantially and require $O(n^2)$ space. As a result, most uses of Lanczos in practice are isolated to use-cases that depend on the extremal parts of the spectrum. The Lanczos method is the primier method remains used in practice.

Recently, Musco et al. @musco2018stability established the stability results regarding bounding the approximation error Paige's $O(n)$-space A27 variant of the Lanczos method for approximating matrix functions via the approximation: 
$ f(A)x approx norm(x) dot Q f(T) e_1 $
In particular, up to a factor of 2, the error of the degree-$k$ Lanczos-based approximation is bounded by the uniform error of the best polynomial approximation to $f$ with degree < $k$. To our knowledge, this was the first matrix function approximation bound for Lanczos in finite-precision. 

// This is, to our knowledge, the first finite precision results justifying the use of Lanczos on such problems (as compared to alternative algorithms, such as the Jacobi Davidson or Chebychev approximations). Indeed, Chen et al. recommend the implementation of Cheychev approximations _via_ Lanczos (cite). Combining this result with the prior two Lemmas, we have our first algorithm and corresponding theorem.


#theorem[Feature extraction complexity][
	Any featurization of the form $op("diag")(f(cal(L)_p))$ can be computed in finite precision in $O(m r)$ time and and $O(min(m, n))$ space. 
]

== Iterative $epsilon$-approximation

For situations where exact computation is infeasible, it is often desirable to have a $(1 plus.minus epsilon)$-approximation scheme. For certain matrix / matrix function combinations, such as matrices with low rank or matrix functions that emphasize the dominant eigenspaces, randomized approximations have proven to be state of the art in terms of performance. 

One approach to construct an $epsilon$-approximation of our designated featurization is to use a randomized Hutchinson-type diagonal estimator, inspired by the trace estimator of the same kind: 

$ r(A) = frac(1, m) sum_(i=1)^m v^((i)) dot.circle A v^((i)) $

where $dot.circle$ denotes the Hadamard (entrywise) product, and $v^((i)) in {-1, +1}^n$ denotes the $i$-th random vector with i.i.d Rademacher entries. It is not hard to show that $r(A)$ is an unbiased estimator the diagonal of matrix, and that this estimator remains unbiased for matrix functions $f(A)$ when the error associated with $v |-> f(A) v$ is uniform.    

Akin to the trace estimator, it's been shown that for any $delta in (0, 1]$ and $m gt.eq 1$, the error of $r(A)$ is upper bounded by: 
$ norm(r^m (A) - op("diag")(A))_2 lt.eq c sqrt(log(2 slash delta) / m) (norm(A)_F^2 - norm(op("diag")(A))_2^2) $
where $c$ is an absolute constant independent of $A$. Thus, to achieve a relative error of $epsilon dot (norm(A)_F^2 - norm(op("diag")(A))_2^2)$, it suffices to perform $m = O(log(1 slash delta) slash epsilon^2 )$ matrix-vector products @dharangutte2023tight. 

For some types of matrix functions, this crude Monte-carlo approach can be more than sufficient for featurization purposes. Nonetheless, as this topic has gained much attention recently, there may be room for improvement. For example, the Hutch++ trace estimator can achieve a multiplicative $(1 plus.minus epsilon)$-approximation of any PSD matrix using just $Omega(1 epsilon log(1 + epsilon)))$ matrix-vector products using deflation techniques @meyer2021hutch. #footnote[In @meyer2021hutch, a lower bound is given showing that any algorithm that accesses PSD matrices via matrix-vector products _non-adaptively_ requires at least $Omega(1 slash epsilon)$ to achieve a $(1 plus.minus epsilon)$-approximation of $op("tr")(A)$ with probability $> 3/4$. We do not know of any such algorithm that achieves an analogous bound for the purpose diagonal estimation.]<fn_diag_est> For more recent updates on this area, see @meyer2021hutch, @dharangutte2023tight and references therein.


// #theorem[
// 	Under mild assumptions, if the function $f$ is analytic in the domain $[lambda_min, lambda_max]$, any signature of the form @eq_diag_mf can be $(1 plus.minus epsilon)$-approximated using on the order of $O()$ matrix-function evaluations $v |-> f(A) v$. 
// ] <thm_mf_approx>



= Results

== Benchmarks 

We implement the Hutchinson, Hutch++, divide-and-conquer, and Lanczos with reorthogonalization on real-world simplicial complex to compare the balance between approximation quality, computation time, and space usage. 





// We implement this lift An implementation of the topological + affinity weighting scheme from 
// 2. Fast downward closure code for restricting to $d$-skeletons
// 3. Three additional hypergraph datasets (1 toy and 2 real)

// > **NOTE:** Two of the three supplied datasets come from Google drive links, which require the `gdown` package to download (we provide these scripts as valid loaders in the pipeline). 

// 5. Proof of concept modeling code taken from the [topomodelx tutorial page](https://pyt-team.github.io/topomodelx/tutorials/index.html)

// The lifting code itself requires the `hirola` package to efficiently compute the $d$-skeleton. This was added as a dependency to the `pyproject.toml`. 

= Conclusion 

#bibliography("references.bib")

// #pagebreak()

#heading(numbering: none)[Appendix]

== Proofs 

#proof([of ])[ // @thm_mf_approx
The proof of the above statement essentially follows from combining three recent works: 
	1. Ubaru's finite-precision results on the stability of the Lanczos method 
	2. The stochastic Hutchinson-style diagonal estimator (applied to matrix functions)
	3. The above proof that the heat kernel reduces to a diagonal
]

#proof([of ])[
	Proof: If the operation $v -> A v$ can be done in $O(nu)$ space, it follows from the 3-term recurrence of the Lanczos method and the fact that combinatorial Laplacian require $O(n)$ space to perform matrix-vector actions that the above signature can be computed in space complexity $O(n)$. 
]

#proof[][
	The proof of the finite-precision follows directly by combining Lemma's () and (), see for more details. 
	To show that the space complexity cannot exceed $O(min(m,n))$, we must show that both
	$ cal(L) = partial_1 partial_1^T + partial_1^T partial_1 $ 
	i.e. that the up and down Laplacians both require at most $O(min(m,n))$-space in their matrix-vector actions. 
	To show this, we recall onnections between the up- and down- Laplacians:
	$ + $ 
	Thus, the rank of either Laplacina is at most $min(m,n)$.
]

