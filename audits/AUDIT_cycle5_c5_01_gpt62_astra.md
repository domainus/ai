# Cycle 5 C5-01 live audit on RTX 5080

**Verdict: FAIL. Hold promotion of the combined change at `596b522`.** All 18 native K = 2²⁵ cases met their individual spectral ceilings. The public SYRK API also fails on a valid contiguous input and misses correct destination rounding on two exact witnesses. The independently implemented MMA rule fits the instruction probes when a truncated zero magnitude encodes as +0. Both supplied model functions disagree with raw hardware bits.

| Severity | Finding and impact | Promotion condition |
|---|---|---|
| High | Native SYRK launches 65,536 programs in CUDA grid-y at K = 2²⁸, n = 1. An ordinary 512 MiB FP16 input fails. | Cap chunks per launch at the device grid-y limit as well as the byte limit. Test both sides. |
| High | The FP64 sum rounds through FP32 in FP16 and BF16 stores. Both signs miss the correctly rounded value by one destination ULP. The historical prototype shares the cast. | Provide single rounding for both destinations, or explicitly revise the contract and quantify the difference. |
| High for model exactness | For 16 NaNs the scalar and batched APIs returned `0x7fc00000` as FP32 while hardware returned `0x7fffffff`. The batched API also missed 304 negative-underflow sign bits. | Check unmodified output words. Preserve the measured NaN payload and return +0 on the tested underflows. |
| Medium | The timing driver compares captured output buffers before either graph has replayed. Its equality flags can pass for divergent graphs. | Replay and synchronize before comparing. Retain the separate input-overwrite replay check. |
| Medium | Canonical validation accepts missing or mislabeled Opus cells, and a documented MMA mismatch after its summaries are updated. The hardware hash check accepts arbitrary hex. | Check cell identities and define the promotion criteria explicitly. Verify evidence and source hashes independently when possible. |
| Medium for historical controls | The C4-03 PolarSwitch hook silently uses native chunking below a custom gate greater than 65,536. | Make the unmodified path honor the hook’s gate throughout the context. |
| Low | Each Opus cell puts native first twice and control first once, with two degrees of freedom for its paired interval. | Qualify speedups by the sampling protocol. Use even rounds for a balanced repeat. |

## Source and artifact

The isolated checkout started clean at `596b5222d0fd96fdb6d6ef6c6e6386464361ba0d`, parent `307fe7a8dc4fa7a9ee7642c26e33c42e6ec8293f`, grandparent `289e6d75fc341ce6476867147a000da3d4704969`. The parent changed four research files and added the benchmark driver, MMA model and C5-01 test module. The child adds exactly `results/cycle5_c5_01_syrk_and_mma.json`. Both passed `git show --check`. Correctness probes and regressions overlapped during parts of the audit. The independent latency probe waited for those jobs to finish.

| Check | Result |
|---|---|
| Artifact SHA-256 | `433e188d198182356c62800a5ad534d1b70b212ebcc7722ef6d038dcb631d9d8` |
| Independent code SHA-256 | `a377cca51facf857ed75b85cb2ffcd3e6ff7903a1991b80dd5d5a697d99d48a1` |
| Source closure | 16 modules. Walked AST imports in Git blobs from the parent and hashed sorted module name, NUL, and exact source bytes. All module names matched. The package initializers contain docstrings only. |
| Run hash and metadata | Independently recomputed `973b7142efe940b0`. Recorded `git_sha` equals the parent, `git_dirty` is false, and the three legacy blob IDs match the grandparent. |
| JSON | The duplicate-key and non-finite-constant rejecting parse passed. A strict `allow_nan=False` serialization and reparse preserved the complete value. `validate_results(payload, canonical=True)` passed. |
| Live host | NVIDIA GeForce RTX 5080, capability (12, 0), driver 615.78.08, Python 3.13, PyTorch 2.14.1+cu130, Triton 3.8.0. |
| Regression suite | 366 passed, 6 skipped, 772 warnings in 1348.84s (0:22:28). The six missing-data tests separately passed using the existing 214 MiB harvested file. No test or input file was changed. |

### Validator and CLI probes

Changing the fourth-smallest native GPU sample in the first timing cell (round 37) by +0.001 ms failed the paired-geometric-mean check. Separate mutations to a median, median ratio, CI endpoint, Opus raw sample, Opus geometric mean, MMA bit count, derived total, dirty flag, code hash, config digest and capability also failed. Moving that non-median sample by one binary64 successor passed the configured tolerance. Scaling both GPU times of the same pair by 1.0001 passed with the original summary. This keeps the ratio and both marginal medians.

Valid-looking fabricated hardware hashes and legacy blob IDs passed. So did an extra module name, timing dtype `fp64`, Opus shape `[2,16]` and grid shape `[0,0]`. Rehashing an all-zero commit and code digest passed. Removing an Opus row or recording an MMA mismatch passed after updating the derived section. The independent Git checks above establish the provenance of this particular artifact. The JSON omits the raw output words needed for independent verification of every hardware digest.

Quick CLI parses rejected the canonical filename, its cross-checkout copy, an alias resolving to it and a reserved sibling name. Canonical mode rejected a custom basename. Canonical mode **accepted** the canonical basename in another directory and its symlink alias. The writer resolves its destination before replacing it. The host guard rejected CPU, injected sm_89 and a real tracked edit in a disposable Git repository. It accepted a clean tree on this device and a tree containing an untracked file. Its Git call uses `--untracked-files=no` on the producing checkout.

## Timings and derived statistics

I recomputed the summaries from 4,800 ordinary GPU/wall durations across 30 cells (40 rounds, two variants, two clocks) and 108 Opus wall durations across 18 cells (three rounds, two variants). All 522 comparisons with stored timing fields and derived bounds agreed within **7.11 × 10⁻¹⁴ absolute**. A second calculation independently rebuilt all **137 scalar leaves** of `derived`. Agreement of aggregate counts verifies the recorded rows, including their NaN normalization.

Ordinary ratios use native/control. The median ratio divides marginal medians. The paired estimate exponentiates the mean within-round log ratio. Its 95% interval uses the sample SD and Student t quantile with df = rounds − 1. In **each** ordinary cell both variants ran 20 times in each position, with 20 native→control and 20 control→native within-round predecessor pairs. Every Opus cell ran AB, BA, AB, so its first-position and ordered-pair counts are 2:1. Each ordinary variant gets a separate flush before measurement. Opus used synchronized wall timings of a native call and a pinned unchunked control.

Across ordinary cells, median native/control GPU ratios were 0.998687–1.001045 for graph replay and 0.989651–1.005669 for eager execution. The corresponding wall ratios were 0.997030–1.000831 and 0.986615–1.004997. Across Opus cells, the inverse ratio (control/native) ranged from **28.506577× to 83.499752×** for marginal medians and **28.394640× to 84.187515×** for paired geometric means. Opus stores wall time only.

<details><summary>All 60 normal timing summaries and all 18 Opus speedups</summary>

Each four-number group is minimum / median / IQR / maximum in milliseconds. IQR uses linearly interpolated P75 minus P25. IQR and sample extrema are additional audit calculations. The remaining four fields give the ratio of medians, paired geometric mean, CI low and CI high. Values were rounded for display after validation. The source computes each statistic from the full-precision JSON numbers.

```text
shape kernel mode clock   native[min/median/IQR/max] control[min/median/IQR/max]  ratio geomean low high
512x2048 ps3_55 graph gpu 0.136352/0.137888/0.000903998/0.143072 0.135456/0.137744/0.001416/0.141824 1.00104540 1.00112009 0.9962733 1.00599046
512x2048 ps3_55 graph wall 0.150511/0.152676/0.0014918/0.161071 0.150222/0.152702/0.00229746/0.165051 0.99983228 0.99710308 0.98941282 1.00485311
512x2048 ps3_55 eager gpu 0.270592/0.28264/0.02784/0.379136 0.268288/0.282672/0.0234/0.425952 0.99988677 0.99332869 0.96789877 1.01942674
512x2048 ps3_55 eager wall 0.292793/0.307642/0.0299579/0.407714 0.289743/0.306112/0.0241633/0.461814 1.00499718 0.99411988 0.96934237 1.01953073
512x2048 ss3_54 graph gpu 0.136192/0.138048/0.00144/0.141184 0.135968/0.137904/0.00124801/0.284704 1.00104419 0.98458285 0.94888087 1.02162813
512x2048 ss3_54 graph wall 0.150281/0.152501/0.00260246/0.162821 0.150112/0.152406/0.00168005/0.304372 1.00062330 0.98825446 0.95380443 1.02394878
512x2048 ss3_54 eager gpu 0.268512/0.273888/0.009824/0.302464 0.267072/0.276752/0.013296/0.303968 0.98965143 0.98747383 0.97486489 1.00024586
512x2048 ss3_54 eager wall 0.290763/0.297062/0.00986253/0.331623 0.289703/0.301092/0.0156965/0.341922 0.98661517 0.98595677 0.97231168 0.99979336
512x2048 ss3_53 graph gpu 0.130752/0.131984/0.00123199/0.134304 0.130656/0.131952/0.000784006/0.13408 1.00024251 0.99946416 0.99667689 1.00225923
512x2048 ss3_53 graph wall 0.144501/0.146161/0.00242336/0.153311 0.144861/0.146596/0.00146293/0.157161 0.99702999 1.00061466 0.99351322 1.00776686
512x2048 ss3_53 eager gpu 0.26112/0.269664/0.00908801/0.304192 0.262944/0.268144/0.00827201/0.28304 1.00566857 1.00505049 0.99516128 1.01503797
512x2048 ss3_53 eager wall 0.283632/0.293162/0.0104951/0.340112 0.285272/0.291817/0.00877742/0.312592 1.00460687 1.00585382 0.9958215 1.0159872
1024x4096 ps3_55 graph gpu 0.513952/0.526672/0.00153604/0.530496 0.516288/0.527024/0.001808/0.53584 0.99933205 0.9990405 0.99684937 1.00123646
1024x4096 ps3_55 graph wall 0.539384/0.541779/0.00147737/0.693405 0.539674/0.542304/0.00209239/0.645466 0.99903099 1.00046821 0.98493992 1.01624131
1024x4096 ps3_55 eager gpu 0.576352/0.58048/0.00264797/0.5904 0.5776/0.581424/0.00462401/0.606656 0.99837637 0.99624684 0.99325764 0.99924504
1024x4096 ps3_55 eager wall 0.592764/0.597774/0.0027474/0.608075 0.593885/0.598484/0.00474241/0.622925 0.99881422 0.99634144 0.99352792 0.99916292
1024x4096 ss3_54 graph gpu 0.50912/0.511856/0.00118399/0.514272 0.51008/0.512336/0.00201599/0.520224 0.99906312 0.99828934 0.99722053 0.99935929
1024x4096 ss3_54 graph wall 0.522924/0.526584/0.00150991/0.530464 0.524994/0.527274/0.00256766/0.536764 0.99869139 0.99818182 0.99667882 0.99968709
1024x4096 ss3_54 eager gpu 0.55072/0.554/0.00224799/0.562976 0.550368/0.553808/0.001864/0.561152 1.00034667 1.0000411 0.99757008 1.00251824
1024x4096 ss3_54 eager wall 0.567514/0.57091/0.00253523/0.579765 0.566625/0.570894/0.00561215/0.579285 1.00002753 0.99867295 0.99589325 1.00146042
1024x4096 ss3_53 graph gpu 0.4912/0.494288/0.00159199/0.50272 0.49216/0.494544/0.00157598/0.497376 0.99948235 1.00057282 0.99912044 1.00202731
1024x4096 ss3_53 graph wall 0.505654/0.509549/0.00189251/0.536394 0.506743/0.509699/0.00198488/0.517434 0.99970582 1.00182595 0.99820158 1.00546349
1024x4096 ss3_53 eager gpu 0.532288/0.5364/0.00349601/0.577952 0.532864/0.53728/0.00457598/0.55376 0.99836211 1.00101755 0.99580156 1.00626086
1024x4096 ss3_53 eager wall 0.548315/0.553384/0.00544335/0.599265 0.549514/0.554524/0.00461272/0.571664 0.99794325 1.00122312 0.99582347 1.00665204
2048x8192 ps3_55 graph gpu 3.07808/3.1072/0.00746393/3.1497 3.09933/3.10698/0.013984/3.29696 1.00007213 0.99677904 0.9925748 1.00100109
2048x8192 ps3_55 graph wall 3.1129/3.12632/0.00931264/3.24644 3.11809/3.1273/0.0142072/3.41099 0.99968667 0.99690189 0.99108248 1.00275547
2048x8192 ps3_55 eager gpu 3.13856/3.14694/0.00582409/3.37958 3.13859/3.14634/0.00374407/3.34493 1.00019319 0.99895035 0.99310757 1.00482752
2048x8192 ps3_55 eager wall 3.15638/3.16905/0.00822783/3.39708 3.1579/3.16662/0.00524061/3.36257 1.00076736 0.99904169 0.99334024 1.00477586
2048x8192 ss3_54 graph gpu 3.02541/3.03811/0.00695199/3.04461 3.02042/3.03602/0.00532788/3.05046 1.00069040 1.0006468 1.0001649 1.00112894
2048x8192 ss3_54 graph wall 3.04099/3.05516/0.00782328/3.06998 3.03603/3.05262/0.00606914/3.07832 1.00083080 1.00066932 1.00006447 1.00127453
2048x8192 ss3_54 eager gpu 3.05923/3.07488/0.00738394/3.28426 3.06595/3.07898/0.012304/3.27075 0.99866972 0.99771582 0.99202919 1.00343505
2048x8192 ss3_54 eager wall 3.08705/3.09562/0.00736245/3.30604 3.08401/3.09851/0.0146672/3.29165 0.99906568 0.99775338 0.99221238 1.00332533
2048x8192 ss3_53 graph gpu 2.93712/2.97202/0.0252801/2.98659 2.93994/2.97043/0.025376/2.98032 1.00053323 1.00086662 1.00023351 1.00150014
2048x8192 ss3_53 graph wall 2.95461/2.98731/0.0248375/3.00264 2.95462/2.98711/0.0256125/2.99554 1.00006692 1.00097707 1.00029581 1.00165879
2048x8192 ss3_53 eager gpu 2.97722/2.98453/0.00387996/3.14659 2.98032/2.984/0.00462401/3.1945 1.00017698 0.99830505 0.99476228 1.00186044
2048x8192 ss3_53 eager wall 2.99553/3.003/0.00697782/3.16513 2.9985/3.00337/0.00485912/3.21369 0.99987511 0.99839615 0.99484337 1.00196163
4096x4096 ps3_55 graph gpu 10.5271/10.5352/0.00752831/10.9699 10.5305/10.5373/0.00460029/10.9704 0.99980415 1.00035769 0.99913095 1.00158594
4096x4096 ps3_55 graph wall 10.5459/10.5568/0.00783819/10.9949 10.5504/10.5596/0.00677007/10.9949 0.99973521 1.00004533 0.99927827 1.00081297
4096x4096 ps3_55 eager gpu 10.4593/10.5647/0.00804043/10.7641 10.4544/10.5656/0.00900793/10.7163 0.99990911 0.9998899 0.99915303 1.00062731
4096x4096 ps3_55 eager wall 10.484/10.5881/0.00871276/10.7864 10.478/10.5881/0.0112851/10.8036 1.00000048 0.99964622 0.99860921 1.0006843
4096x4096 ss3_54 graph gpu 9.83306/9.93462/0.00614381/10.1155 9.86314/9.93632/0.00639224/10.3113 0.99982935 0.99771605 0.99507033 1.0003688
4096x4096 ss3_54 graph wall 9.85101/9.9552/0.00857993/10.1377 9.8809/9.95781/0.00981503/10.3309 0.99973788 0.99773556 0.9950816 1.00039661
4096x4096 ss3_54 eager gpu 9.84771/9.95264/0.00633621/10.4715 9.84717/9.95053/0.00898385/10.1721 1.00021219 1.00112558 0.99940774 1.00284637
4096x4096 ss3_54 eager wall 9.86946/9.97421/0.00775576/10.4943 9.86791/9.97303/0.0104954/10.1939 1.00011882 1.00111375 0.99938224 1.00284826
4096x4096 ss3_53 graph gpu 9.3105/9.32117/0.00492835/9.49091 9.31482/9.32082/0.00618386/9.7233 1.00003775 0.99941978 0.99818458 1.00065652
4096x4096 ss3_53 graph wall 9.33197/9.34274/0.00664958/9.6686 9.33332/9.34252/0.00621332/9.74901 1.00002409 0.99986244 0.9993761 1.00034902
4096x4096 ss3_53 eager gpu 9.33242/9.34373/0.00638366/9.91478 9.33466/9.34318/0.00730371/9.3568 1.00005823 1.00340387 0.9996146 1.0072075
4096x4096 ss3_53 eager wall 9.34998/9.36413/0.00751269/9.93326 9.35245/9.36465/0.00853994/9.37519 0.99994447 1.00334554 0.99957696 1.00712832
4x1024x4096 ps3_55 graph gpu 1.85005/1.85642/0.00455999/1.87488 1.85056/1.8588/0.00419202/1.87443 0.99871745 0.99897117 0.99814476 0.99979827
4x1024x4096 ps3_55 graph wall 1.86489/1.87214/0.00506931/1.89167 1.86625/1.87507/0.00434334/1.89094 0.99843953 0.99928514 0.99841145 1.00015959
4x1024x4096 ps3_55 eager gpu 1.88058/1.88662/0.00436792/2.02 1.8792/1.88517/0.004792/4.46774 1.00077233 0.98085673 0.93881302 1.02478331
4x1024x4096 ps3_55 eager wall 1.89759/1.90443/0.00479765/2.03704 1.89853/1.90345/0.00640983/4.48608 1.00051479 0.98089708 0.93913386 1.0245175
4x1024x4096 ss3_54 graph gpu 1.81571/1.82536/0.00486404/1.83859 1.81696/1.82776/0.00567994/1.83424 0.99868690 0.99920851 0.99843877 0.99997885
4x1024x4096 ss3_54 graph wall 1.83144/1.84167/0.0069557/1.8618 1.83251/1.84468/0.00751595/1.85189 0.99836864 0.99914518 0.99832728 0.99996374
4x1024x4096 ss3_54 eager gpu 1.85942/1.86693/0.00691202/1.97965 1.79197/1.86757/0.00697598/1.89789 0.99965732 1.0027763 0.99920319 1.00636219
4x1024x4096 ss3_54 eager wall 1.87727/1.88734/0.0098134/2.01739 1.87544/1.88847/0.00599265/1.93631 0.99940161 1.00172963 0.99911243 1.00435369
4x1024x4096 ss3_53 graph gpu 1.75952/1.77131/0.00575995/1.78509 1.76122/1.77174/0.00566408/1.78518 0.99975616 0.99962141 0.99868296 1.00056075
4x1024x4096 ss3_53 graph wall 1.77913/1.79502/0.00767986/1.8136 1.77735/1.79544/0.0101202/1.81203 0.99976885 0.9998151 0.99851295 1.00111895
4x1024x4096 ss3_53 eager gpu 1.8017/1.80992/0.00539997/1.8231 1.80051/1.80867/0.00511199/1.82899 1.00069001 1.00001361 0.99890991 1.00111854
4x1024x4096 ss3_53 eager wall 1.81986/1.83167/0.00632077/1.84531 1.82122/1.83051/0.00739505/1.8584 1.00063362 0.99968712 0.9983761 1.00099987

shape kernel dtype        native[min/median/IQR/max] control[min/median/IQR/max]  speedup geomean low high
4x33554432 ps3_55 fp16 6.03666/6.07274/0.0554005/6.14746 174.896/176.296/0.700435/176.297 29.03068118 28.89328244 28.42674043 29.36748138
4x33554432 ps3_55 bf16 6.0246/6.14559/0.0696555/6.16391 174.962/175.384/0.860202/176.683 28.53812131 28.74704015 27.52395082 30.02448023
4x33554432 ss3_54 fp16 6.05186/6.11973/0.0502155/6.15229 176.517/177.126/0.304436/177.126 28.94336827 28.96663493 28.25781417 29.6932358
4x33554432 ss3_54 bf16 6.0519/6.1492/0.0598955/6.17169 175.726/175.732/0.306982/176.34 28.57802287 28.72820181 27.85645743 29.62722671
4x33554432 ss3_53 fp16 6.03453/6.13731/0.0522655/6.13906 173.968/174.954/1.61893/177.206 28.5065773 28.73311331 28.22769712 29.24757896
4x33554432 ss3_53 bf16 6.15566/6.15676/0.0457455/6.24715 175.059/175.858/0.502394/176.064 28.56333887 28.39463996 27.91666424 28.88079936
33554432x2 ps3_55 fp16 7.79833/7.90633/0.070911/7.94015 651.876/655.082/5.27013/662.416 82.85542451 83.29049875 80.73382451 85.9281376
33554432x2 ps3_55 bf16 8.90381/9.00207/0.180061/9.26393 658.655/660.751/1.17826/661.011 73.39983895 72.90044835 68.97053208 77.05428983
33554432x2 ss3_54 fp16 7.91408/7.94411/0.100016/8.11411 660.146/662.921/2.47425/665.094 83.44809217 82.94038061 80.06296278 85.92121122
33554432x2 ss3_54 bf16 8.88146/8.96668/0.0681106/9.01768 661.735/665.289/6.64619/675.027 74.19570133 74.51922454 71.42224301 77.75049609
33554432x2 ss3_53 fp16 7.79741/8.12552/0.171931/8.14127 665.458/678.479/8.11094/681.679 83.49975241 84.18751473 81.73717741 86.71130888
33554432x2 ss3_53 bf16 8.88167/8.96388/0.0485304/8.97873 663.359/666.92/1.95692/667.272 74.40083522 74.46861791 73.09498061 75.86806929
1x33554432 ps3_55 fp16 4.63225/4.64548/0.104616/4.84148 168.948/175.156/3.35859/175.666 37.70472617 36.8147909 34.95672844 38.77161535
1x33554432 ps3_55 bf16 4.53045/4.61966/0.0535055/4.63747 166.891/167.397/1.11117/169.113 36.2359119 36.51258373 35.58108243 37.46847144
1x33554432 ss3_54 fp16 4.5271/4.64777/0.0685154/4.66413 169.468/169.816/0.611585/170.691 36.53708764 36.85364492 35.39826129 38.368866
1x33554432 ss3_54 bf16 4.53292/4.61353/0.0433604/4.61964 175.73/177.666/1.17417/178.079 38.50985721 38.60839003 37.12128371 40.15507094
1x33554432 ss3_53 fp16 4.52903/4.60447/0.570809/5.67064 176.622/177.087/0.372468/177.367 38.45986462 36.06590005 26.29399311 49.46944122
1x33554432 ss3_53 bf16 4.58326/4.59359/0.054235/4.69173 175.2/175.275/1.21915/177.638 38.15642784 38.08117094 37.59475197 38.57388344
```

</details>

### Independent short-K latency

I loaded all three legacy Blackwell modules from the grandparent, warmed the implementations, shared native autotuning choices with legacy and compared output words. Six shape/circuit pairs matched. I measured 24 alternating rounds per cell for graph replay and eager calls, with a 4×L2 flush before each invocation. The table gives native/legacy ratios and paired geometric 95% intervals. This was one desktop GPU run. The median GPU ratios ranged from 0.99677 to 1.00812. Of the 24 nominal GPU/wall intervals, one excluded parity. The 2,048 × 8,192 SS3_53 graph GPU ratio was 1.00139 with a paired interval of [1.00045, 1.00193]. Two fresh 64-round repeats reversed graph capture order. Their median GPU ratios were 1.00020 and 1.00010, with intervals [0.99908, 1.00295] and [0.99957, 1.00052]. Both repeat wall intervals also included parity. These runs found no repeatable short-K slowdown.

| Shape | Circuit | Mode | Native / legacy GPU medians (ms) | GPU ratio [paired 95% CI] | Native / legacy wall medians (ms) | Wall ratio [paired 95% CI] |
|---|---|---|---:|---:|---:|---:|
| 512 × 2048 | ps3_55 | graph | 0.12866 / 0.12782 | 1.0065 [0.9967, 1.0069] | 0.14176 / 0.14125 | 1.0036 [0.9845, 1.0068] |
| 512 × 2048 | ps3_55 | eager | 0.23230 / 0.23043 | 1.0081 [0.9460, 1.0350] | 0.25237 / 0.25151 | 1.0034 [0.9413, 1.0323] |
| 512 × 2048 | ss3_54 | graph | 0.12966 / 0.12958 | 1.0006 [0.9959, 1.0029] | 0.14202 / 0.14209 | 0.9995 [0.9852, 1.0083] |
| 512 × 2048 | ss3_54 | eager | 0.22226 / 0.22298 | 0.9968 [0.9901, 1.0321] | 0.24282 / 0.24255 | 1.0011 [0.9904, 1.0300] |
| 512 × 2048 | ss3_53 | graph | 0.12416 / 0.12408 | 1.0006 [0.9961, 1.0046] | 0.13655 / 0.13688 | 0.9976 [0.9926, 1.0044] |
| 512 × 2048 | ss3_53 | eager | 0.21595 / 0.21629 | 0.9984 [0.9762, 1.0555] | 0.23626 / 0.23688 | 0.9974 [0.9750, 1.0540] |
| 2048 × 8192 | ps3_55 | graph | 3.09219 / 3.09096 | 1.0004 [0.9970, 1.0097] | 3.10705 / 3.10533 | 1.0006 [0.9972, 1.0098] |
| 2048 × 8192 | ps3_55 | eager | 3.13560 / 3.13459 | 1.0003 [0.9993, 1.0010] | 3.15835 / 3.15505 | 1.0010 [0.9990, 1.0012] |
| 2048 × 8192 | ss3_54 | graph | 3.01152 / 3.01258 | 0.9996 [0.9995, 1.0013] | 3.02616 / 3.02781 | 0.9995 [0.9994, 1.0013] |
| 2048 × 8192 | ss3_54 | eager | 3.06090 / 3.05648 | 1.0014 [0.9996, 1.0012] | 3.07823 / 3.07313 | 1.0017 [0.9994, 1.0011] |
| 2048 × 8192 | ss3_53 | graph | 2.92859 / 2.92453 | 1.0014 [1.0004, 1.0019] | 2.94576 / 2.94412 | 1.0006 [0.9999, 1.0016] |
| 2048 × 8192 | ss3_53 | eager | 2.96533 / 2.96258 | 1.0009 [0.9968, 1.0105] | 2.98214 / 2.97860 | 1.0012 [0.9970, 1.0106] |

The canonical graph flag has a separate defect. On production PS3_55 at 512 × 2048, immediately after capture, the two output buffers matched each other and **neither matched eager**. After replay both matched eager bitwise. In a deterministic control, two graphs recorded writes of different constants. The pre-replay outputs were both zero. Replay gave 1 and 2. The independent replay and input-overwrite checks establish actual output behavior.

## Long-K kernel and circuit checks

PolarSwitch dispatches its unsketched long-side Gram above 65,536. SchattenSwitch dispatches its segment-1 Gram, computes one squared Frobenius partial from the stored matrix, and passes 4,096 as the accumulation length to the bound. Later segments independently dispatch their Grams. The sketch path keeps its original launch. In the default C4-03 hook the raised Schatten gate exposes both original launches to the replacement functions. Exiting the context restores the hooks and gate.

- All six K boundaries passed in both orientations. Inputs used offset views with a storage step of two, two batches and sparse independent FP64 reference Grams. All three output dtypes matched. Circuit comparisons covered both orientations, both working dtypes and three circuits against the default historical hook (72 pairs). Below the default gate the hook delegates to the native single-launch path. The separate regression suite compares pre-C5-01 source directly. Vector-shaped inputs passed in both orientations.
- At n = 512, K = 70,000, the production cap permits **256** full chunk Grams per launch and the input has only 17. A 4 MiB test cap forced groups `4,4,4,4,1`, then a remainder. A separate n = 2,048 case forced groups of 16 and 1 under the default 256 MiB cap, then a remainder. All three outputs matched the sparse reference exactly. The tested Grams had exact transpose symmetry under int16/int32 views, including both failing rounding cases.
- Row, batch and mirrored-store offsets widen before multiplication. The K loop advances pointers. In-tile column multiplication and `BK * sxk` require the guard for i32 strides. At stride 2²⁴ − 1 the compiler emitted i32 and the guard accepted. It rejected 2²⁴ and 2³¹ − 1. It accepted 2³¹ and 2³², for which the compiler emitted i64. The largest accepted probes used a singleton dimension to check specialization. A separate two-column GPU view exercised the rejection on a real strided tensor.
- In mixed batches the normal matrix stayed valid. Both Schatten variants returned exact zeros for zero, tiny, NaN and Inf inputs. PolarSwitch returned zeros for zero and tiny and propagated non-finites on NaN/Inf, consistent with the receipt. Explicit `sigma_hat` and Frobenius scaling matched the historical hook on both Schatten variants.
- Warm SYRK, PolarSwitch, SchattenSwitch and Frobenius scaling passed sync-debug error mode and showed zero D2H copies in the profiler. Explicit `sigma_hat=.9` triggered sync-debug in documented host setup. Its trace showed one HtoD copy and zero D2H copies. Three input overwrites replayed bitwise equal to eager for all six circuit/dtype combinations and standalone SYRK. Distinct overwritten inputs produced distinct outputs.

**Grid limit.** At n = 1, K = 268,435,456 and chunk length 4,096 the byte cap admits 65,536 partials in one launch. The driver reports `MAX_GRID_DIM_Y = 65535`. The public API failed with `Triton Error [CUDA]: invalid argument` on both an expanded view and a contiguous FP16 input. A pinned production kernel on a contiguous all-ones input failed with a cap of 65,536 and returned exactly 268,435,456 with a cap of 65,535.

**Double rounding.** Use two opposite rows, one squared term per chunk and K = 65,537. The positive row has the values below starting at 0, 4,096, 8,192 and, where needed, 12,288. The other entries are zero. The partials and FP64 sum are exact. Each sum lies just above the midpoint between 1.0 and the next destination value. The FP16 witness sums to 1 + 2⁻¹¹ + 2⁻²⁶. The BF16 witness sums to 1 + 2⁻⁸ + 2⁻²⁶. An intermediate FP32 rounding erases the last term in both cases.

| Destination | Nonzero FP16 inputs in distinct chunks | Positive / negative result bits | Correct single-rounding bits |
|---|---|---|---|
| FP16 | 1, 2⁻⁶, 2⁻⁶, 2⁻¹³ | `3c00 / bc00` | `3c01 / bc01` |
| BF16 | 1, 2⁻⁴, 2⁻¹³ | `3f80 / bf80` | `3f81 / bf81` |
| FP32 | 1, 2⁻¹², 2⁻²⁴ | `3f800001 / bf800001` | `3f800001 / bf800001` |

**Historical hook.** With K = 100,000, seed 983 and `ChunkedGram(4096,131072)`, the hook recorded zero launches yet returned native chunked entry 99,121.3359375. The ungated single-launch result was 99,060.8671875. Below the custom gate the hook delegates to PolarSwitch’s original function, which independently applies the 65,536 gate. The canonical comparison used matching gates.

### Independent Opus L6 Gram eigenvalues

I regenerated all three canonical seed-0 inputs. For each native output I formed short-side FP64 output and cross Grams in 349,525-column blocks, then used NumPy on CPU for the eigendecomposition and reference polar factor. This differs from the producer in block length, reduction and eigensolver. All 18 cases stayed below their own design ceilings. Maximum absolute discrepancy across stored minimum singular value, maximum singular value and relative polar error was **3.27 × 10⁻¹³**. The persisted grid has 654/654 cells below their individual ceilings and 10 above the separate common 1.08 ceiling.

| Shape | Circuit | Work dtype | σ min | σ max | Relative polar error | Design ceiling |
|---|---|---|---:|---:|---:|---:|
| 4 × 33554432 | ps3_55 | fp16 | 0.926994876 | 0.928481114 | 0.072210576 | 1.080953841 |
| 4 × 33554432 | ps3_55 | bf16 | 0.923098402 | 0.924004213 | 0.076456401 | 1.080953841 |
| 4 × 33554432 | ss3_54 | fp16 | 1.034350790 | 1.036842041 | 0.035554514 | 1.079825873 |
| 4 × 33554432 | ss3_54 | bf16 | 1.000459504 | 1.006215761 | 0.004757402 | 1.079825873 |
| 4 × 33554432 | ss3_53 | fp16 | 1.009937642 | 1.011915950 | 0.011051999 | 1.087545088 |
| 4 × 33554432 | ss3_53 | bf16 | 0.986665372 | 0.990248023 | 0.011685002 | 1.087545088 |
| 33554432 × 2 | ps3_55 | fp16 | 0.922337273 | 0.922766935 | 0.077448807 | 1.080953841 |
| 33554432 × 2 | ps3_55 | bf16 | 0.920919289 | 0.921101260 | 0.079021826 | 1.080953841 |
| 33554432 × 2 | ss3_54 | fp16 | 0.996557388 | 0.997439024 | 0.003049869 | 1.079825873 |
| 33554432 × 2 | ss3_54 | bf16 | 1.036119006 | 1.036360447 | 0.036318782 | 1.079825873 |
| 33554432 × 2 | ss3_53 | fp16 | 1.012655148 | 1.012848335 | 0.012755947 | 1.087545088 |
| 33554432 × 2 | ss3_53 | bf16 | 0.996692406 | 0.997147400 | 0.003879757 | 1.087545088 |
| 1 × 33554432 | ps3_55 | fp16 | 1.016177323 | 1.016177323 | 0.016220457 | 1.080953841 |
| 1 × 33554432 | ps3_55 | bf16 | 0.993323088 | 0.993323088 | 0.007076715 | 1.080953841 |
| 1 × 33554432 | ss3_54 | fp16 | 0.958751090 | 0.958751090 | 0.041251288 | 1.079825873 |
| 1 × 33554432 | ss3_54 | bf16 | 1.028102917 | 1.028102917 | 0.028204112 | 1.079825873 |
| 1 × 33554432 | ss3_53 | fp16 | 0.914465781 | 0.914465781 | 0.085534908 | 1.087545088 |
| 1 × 33554432 | ss3_53 | bf16 | 0.970436160 | 0.970436160 | 0.029653928 | 1.087545088 |

## Raw MMA arithmetic

The independent integer oracle decodes operand and accumulator words into signed significands and stored field exponents. It aligns each term with integer shifts, adds once per k16 instruction, encodes FP32 and feeds that word into the next instruction. It handles invalid operations, both infinity signs and the normal/subnormal boundary explicitly. The final rule emits +0 when truncation leaves zero magnitude. A preliminary signed-underflow interpretation missed 304 cases. The oracle calls no producer arithmetic. The hardware adapter passes raw 16-bit operands and 32-bit accumulators to the compiled tile kernel, preserving signaling-NaN payloads through the load. The comparison bypasses producer NaN normalization.

Both PTX forms were emitted: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` and `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. On fresh direct-instruction cases the independent rule matched **14,954/14,954**. The scalar API, encoded as FP32, matched **14,938/14,954**. The CPU-vectorized API matched **14,634/14,954**. Repeating all 320 failing rows with the vectorized model resident on CUDA reproduced the same 16 NaN payloads and 304 negative zeros. The separate all-config test matched **1,782/1,782** raw FP32 Gram entries and **1,782/1,782** fused stores (81 entries × 11 configurations × 2 dtypes). Off-diagonal dots use distinct rows. Each matrix count includes mirrored entries.

| Minimal witness | A, B, C words, all omitted lanes zero | Independent / scalar / vector / hardware |
|---|---|---|
| Invalid FP16 product | A[0] = `7c00`, B[0] = `0000`, C = `00000000` | `7fffffff / 7fc00000 / 7fc00000 / 7fffffff` |
| BF16 negative underflow | A[0] = `9780`, B[0] = `1780`, C = `00000000` | `00000000 / 00000000 / 80000000 / 00000000` |

The BF16 values are −2⁻⁸⁰ and +2⁻⁸⁰. The producer’s `_edge_cases` and `_count` replace every NaN with `0x7fffffff` before comparing or hashing bits. The persisted NaN tests therefore cannot distinguish the scalar’s `math.nan` or the vector’s quiet NaN from the device payload. Its aggregate 1,027,034/1,027,034 recorded dots reproduces from the rows.

Live cases covered signed zeros, subnormal operands and accumulators, both transitions across minimum FP32 normal, FLT_MAX, same-sign carries, cancellation, opposite-sign infinities, intermediate overflow followed by cancellation, and non-symmetric chains through K = 4,096. With FLT_MAX as C, a BF16 pair +2¹⁴⁰, −2¹⁴⁰ and a unit product returned finite `0x7f7ff800` after alignment truncated low bits of C. The original four-term FP16 witness returned `0x3d1631fa`. The five-term BF16 reconstruction returned `0x3d313cb9`. A separate production launch reproduced each witness in 11/11 configurations. The C4-03 report gives only BF16 result values, so the original five lane values could not be verified.

<details><summary>All live instruction and config bit counts</summary>

M rows give dtype, case, total, independent oracle matches, scalar matches and vector matches. CFG rows give production configuration, raw matches, fused matches and total. BITS rows give hardware / independent / scalar / vector for each targeted input.

```text
BITS fp16 negzero 00000000 00000000 00000000 00000000
M fp16 negzero 1 1 1 1
BITS fp16 cancel 00000000 00000000 00000000 00000000
M fp16 cancel 1 1 1 1
BITS fp16 zero_negative_product 00000000 00000000 00000000 00000000
M fp16 zero_negative_product 1 1 1 1
BITS fp16 carry 4283df7e 4283df7e 4283df7e 4283df7e
M fp16 carry 1 1 1 1
BITS fp16 carry_cancel 00000000 00000000 00000000 00000000
M fp16 carry_cancel 1 1 1 1
BITS fp16 maxacc 7f7fffff 7f7fffff 7f7fffff 7f7fffff
M fp16 maxacc 1 1 1 1
BITS fp16 maxacc_neg ff7fffff ff7fffff ff7fffff ff7fffff
M fp16 maxacc_neg 1 1 1 1
BITS fp16 max16 517fc004 517fc004 517fc004 517fc004
M fp16 max16 1 1 1 1
BITS fp16 infx0 7fffffff 7fffffff 7fc00000 7fc00000
M fp16 infx0 1 1 0 0
BITS fp16 mixedinf 7fffffff 7fffffff 7fc00000 7fc00000
M fp16 mixedinf 1 1 0 0
BITS fp16 neg_inf ff800000 ff800000 ff800000 ff800000
M fp16 neg_inf 1 1 1 1
BITS fp16 pos_inf 7f800000 7f800000 7f800000 7f800000
M fp16 pos_inf 1 1 1 1
BITS fp16 nan_a1 7fffffff 7fffffff 7fc00000 7fc00000
M fp16 nan_a1 1 1 0 0
BITS fp16 nan_b1 7fffffff 7fffffff 7fc00000 7fc00000
M fp16 nan_b1 1 1 0 0
BITS fp16 nan_a512 7fffffff 7fffffff 7fc00000 7fc00000
M fp16 nan_a512 1 1 0 0
BITS fp16 nan_b512 7fffffff 7fffffff 7fc00000 7fc00000
M fp16 nan_b512 1 1 0 0
BITS fp16 snan_c 7fffffff 7fffffff 7fc00000 7fc00000
M fp16 snan_c 1 1 0 0
BITS fp16 qnan_c 7fffffff 7fffffff 7fc00000 7fc00000
M fp16 qnan_c 1 1 0 0
BITS fp16 C4_witness 3d1631fa 3d1631fa 3d1631fa 3d1631fa
M fp16 C4_witness 1 1 1 1
M fp16 normal 1024 1024 1024 1024
M fp16 full_finite 1024 1024 1024 1024
M fp16 both_sub 1024 1024 1024 1024
M fp16 sub_x_normal 1024 1024 1024 1024
M fp16 fp32_subacc 1024 1024 1024 1024
M fp16 near_fp32_min 1024 1024 1024 1024
M fp16 chain16 32 32 32 32
M fp16 chain32 32 32 32 32
M fp16 chain64 32 32 32 32
M fp16 chain128 32 32 32 32
M fp16 chain256 32 32 32 32
M fp16 chain512 32 32 32 32
M fp16 chain1024 32 32 32 32
M fp16 chain2048 32 32 32 32
M fp16 chain4096 32 32 32 32
CFG fp16 {'BLOCK': 128, 'BK': 64, 'num_warps': 8, 'num_stages': 3} 81 81 81
CFG fp16 {'BLOCK': 128, 'BK': 64, 'num_warps': 4, 'num_stages': 3} 81 81 81
CFG fp16 {'BLOCK': 128, 'BK': 32, 'num_warps': 8, 'num_stages': 4} 81 81 81
CFG fp16 {'BLOCK': 128, 'BK': 32, 'num_warps': 4, 'num_stages': 4} 81 81 81
CFG fp16 {'BLOCK': 128, 'BK': 64, 'num_warps': 8, 'num_stages': 2} 81 81 81
CFG fp16 {'BLOCK': 64, 'BK': 64, 'num_warps': 4, 'num_stages': 4} 81 81 81
CFG fp16 {'BLOCK': 64, 'BK': 64, 'num_warps': 4, 'num_stages': 3} 81 81 81
CFG fp16 {'BLOCK': 64, 'BK': 32, 'num_warps': 4, 'num_stages': 5} 81 81 81
CFG fp16 {'BLOCK': 64, 'BK': 128, 'num_warps': 4, 'num_stages': 3} 81 81 81
CFG fp16 {'BLOCK': 32, 'BK': 64, 'num_warps': 4, 'num_stages': 4} 81 81 81
CFG fp16 {'BLOCK': 32, 'BK': 128, 'num_warps': 4, 'num_stages': 3} 81 81 81
BITS bf16 negzero 00000000 00000000 00000000 00000000
M bf16 negzero 1 1 1 1
BITS bf16 cancel 00000000 00000000 00000000 00000000
M bf16 cancel 1 1 1 1
BITS bf16 zero_negative_product 00000000 00000000 00000000 00000000
M bf16 zero_negative_product 1 1 1 1
BITS bf16 carry 4282fffc 4282fffc 4282fffc 4282fffc
M bf16 carry 1 1 1 1
BITS bf16 carry_cancel 00000000 00000000 00000000 00000000
M bf16 carry_cancel 1 1 1 1
BITS bf16 maxacc 7f7fffff 7f7fffff 7f7fffff 7f7fffff
M bf16 maxacc 1 1 1 1
BITS bf16 maxacc_neg ff7fffff ff7fffff ff7fffff ff7fffff
M bf16 maxacc_neg 1 1 1 1
BITS bf16 max16 7f800000 7f800000 7f800000 7f800000
M bf16 max16 1 1 1 1
BITS bf16 infx0 7fffffff 7fffffff 7fc00000 7fc00000
M bf16 infx0 1 1 0 0
BITS bf16 mixedinf 7fffffff 7fffffff 7fc00000 7fc00000
M bf16 mixedinf 1 1 0 0
BITS bf16 neg_inf ff800000 ff800000 ff800000 ff800000
M bf16 neg_inf 1 1 1 1
BITS bf16 pos_inf 7f800000 7f800000 7f800000 7f800000
M bf16 pos_inf 1 1 1 1
BITS bf16 nan_a1 7fffffff 7fffffff 7fc00000 7fc00000
M bf16 nan_a1 1 1 0 0
BITS bf16 nan_b1 7fffffff 7fffffff 7fc00000 7fc00000
M bf16 nan_b1 1 1 0 0
BITS bf16 nan_a64 7fffffff 7fffffff 7fc00000 7fc00000
M bf16 nan_a64 1 1 0 0
BITS bf16 nan_b64 7fffffff 7fffffff 7fc00000 7fc00000
M bf16 nan_b64 1 1 0 0
BITS bf16 snan_c 7fffffff 7fffffff 7fc00000 7fc00000
M bf16 snan_c 1 1 0 0
BITS bf16 qnan_c 7fffffff 7fffffff 7fc00000 7fc00000
M bf16 qnan_c 1 1 0 0
BITS bf16 overflow_then_cancel 7f7ff800 7f7ff800 7f7ff800 7f7ff800
M bf16 overflow_then_cancel 1 1 1 1
BITS bf16 negative_tiny 00000000 00000000 00000000 80000000
M bf16 negative_tiny 1 1 1 0
BITS bf16 minnormal_from_below 00800000 00800000 00800000 00800000
M bf16 minnormal_from_below 1 1 1 1
BITS bf16 subnormal_from_above 007fffff 007fffff 007fffff 007fffff
M bf16 subnormal_from_above 1 1 1 1
BITS bf16 C4_witness 3d313cb9 3d313cb9 3d313cb9 3d313cb9
M bf16 C4_witness 1 1 1 1
M bf16 normal 1024 1024 1024 1024
M bf16 full_finite 1024 1024 1024 1024
M bf16 both_sub 1024 1024 1024 1024
M bf16 sub_x_normal 1024 1024 1024 1024
M bf16 fp32_subacc 1024 1024 1024 1024
M bf16 near_fp32_min 1024 1024 1024 1024
M bf16 chain16 32 32 32 32
M bf16 chain32 32 32 32 32
M bf16 chain64 32 32 32 32
M bf16 chain128 32 32 32 32
M bf16 chain256 32 32 32 32
M bf16 chain512 32 32 32 32
M bf16 chain1024 32 32 32 32
M bf16 chain2048 32 32 32 32
M bf16 chain4096 32 32 32 32
CFG bf16 {'BLOCK': 128, 'BK': 64, 'num_warps': 8, 'num_stages': 3} 81 81 81
CFG bf16 {'BLOCK': 128, 'BK': 64, 'num_warps': 4, 'num_stages': 3} 81 81 81
CFG bf16 {'BLOCK': 128, 'BK': 32, 'num_warps': 8, 'num_stages': 4} 81 81 81
CFG bf16 {'BLOCK': 128, 'BK': 32, 'num_warps': 4, 'num_stages': 4} 81 81 81
CFG bf16 {'BLOCK': 128, 'BK': 64, 'num_warps': 8, 'num_stages': 2} 81 81 81
CFG bf16 {'BLOCK': 64, 'BK': 64, 'num_warps': 4, 'num_stages': 4} 81 81 81
CFG bf16 {'BLOCK': 64, 'BK': 64, 'num_warps': 4, 'num_stages': 3} 81 81 81
CFG bf16 {'BLOCK': 64, 'BK': 32, 'num_warps': 4, 'num_stages': 5} 81 81 81
CFG bf16 {'BLOCK': 64, 'BK': 128, 'num_warps': 4, 'num_stages': 3} 81 81 81
CFG bf16 {'BLOCK': 32, 'BK': 64, 'num_warps': 4, 'num_stages': 4} 81 81 81
CFG bf16 {'BLOCK': 32, 'BK': 128, 'num_warps': 4, 'num_stages': 3} 81 81 81
M bf16 underflow 512 512 512 218
M bf16 sub_boundary 512 512 512 503
M bf16 overflow 512 512 512 512
M bf16 cancellation 512 512 512 512
HIST fp16 0000 0000 0000 0000 0000 0000 16a8 0000 0000 0000 0000 3220 0000 0000 1d93 0de9 C=00000000 0x3d1631fa 11/11
HIST bf16 0000 0000 0000 0000 0000 0000 0000 0000 3e55 0000 399e 385e 0000 39a9 3af0 0000 C=00000000 0x3d313cb9 11/11
```

</details>

<details><summary>All 320 raw-bit counterexamples</summary>

Each record gives dtype, case, zero-based index, the 16 words of A, the 16 words of B, then C / independent / scalar / vector / hardware. All words are hexadecimal.

```text
fp16 | infx0 | 0 | 7c00 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
fp16 | mixedinf | 0 | 7c00 fc00 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 3c00 3c00 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
fp16 | nan_a1 | 0 | 7c01 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 3c00 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
fp16 | nan_b1 | 0 | 3c00 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 7c01 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
fp16 | nan_a512 | 0 | 7e00 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 3c00 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
fp16 | nan_b512 | 0 | 3c00 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 7e00 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
fp16 | snan_c | 0 | 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 7f800001 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
fp16 | qnan_c | 0 | 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 7fc00000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
bf16 | infx0 | 0 | 7f80 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
bf16 | mixedinf | 0 | 7f80 ff80 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 3f80 3f80 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
bf16 | nan_a1 | 0 | 7f81 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 3f80 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
bf16 | nan_b1 | 0 | 3f80 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 7f81 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
bf16 | nan_a64 | 0 | 7fc0 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 3f80 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
bf16 | nan_b64 | 0 | 3f80 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 7fc0 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
bf16 | snan_c | 0 | 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 7f800001 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
bf16 | qnan_c | 0 | 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 7fc00000 | 7fffffff | 7fc00000 | 7fc00000 | 7fffffff
bf16 | negative_tiny | 0 | 9780 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 1780 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 1 | 1290 1865 922d 1325 991e 19af 16dc 175f 148d 112d 9974 9af8 13f4 1562 10ee 9189 | 1cb0 9616 1c4b 1ac0 1610 9229 9609 156a 98dc 1b1c 1583 931b 99f4 1770 9ce9 1ab9 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 4 | 1861 9705 181d 152b 17b5 9979 9262 98fb 9791 95a0 944a 9863 19b5 8f2a 1843 915d | 90f4 1237 9369 9397 912a 0e25 196a 1288 106c 157d 1546 90ef 11d1 183d 8f67 99ec | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 5 | 933f 133c 188c 968c 9ac8 9afc 9733 9b36 9190 98fb 9626 97eb 9514 1916 94f1 9779 | 1c5c 9c73 1698 18ed 96fc 155d 9833 15a7 9e33 194b 1b68 1728 9c2f 1642 9b76 17c2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 6 | 1c00 9799 1794 1515 948a 9b89 966e 13d2 1165 9ab5 1ba9 1741 982d 17cc 9519 910b | 929d 9530 9543 993d 987c 13a1 9839 18c4 1d2f 14ae 9277 976b 93f6 93cd 16eb 9a15 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 8 | 1828 9af1 132a 1c04 9bb4 95c2 1566 17d7 95a8 0fc0 9a9e 1356 16a1 1807 99e7 9253 | 9429 92ef 990c 1275 9276 1840 961c 16dc 1964 1c22 1404 9937 95b7 13a8 12cf 19b5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 9 | 18b3 9b6d 1bd6 1aa0 1299 130e 9ab0 1449 9b42 1697 9ccf 9989 9ad3 99ed 1b06 1a48 | 16b2 90a3 92be 11f0 9a33 9af5 9201 98ed 1444 18ba 137f 165b 14c6 1368 935a 1343 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 10 | 9c13 9d4d 141f 169f 1ac5 1c29 124d 1c30 1122 13ad 16e3 1632 16b5 1b8a 14ea 1dee | 9559 94f7 1a89 1978 9534 92a9 1d0e 93c0 9d64 1b61 9b6d 18df 1a59 9457 9dd7 946e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 11 | 926a 199b 9016 9395 1917 1932 186d 9113 965e 9a65 152d 19e2 9165 9b58 1803 947c | 184c 122d 1b22 1887 1471 934f 934c 9ce7 169e 136d 98de 934f 9912 9207 12ae 972d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 13 | 13a7 19a8 11ed 122d 8f29 9ac4 18c5 94bc 1840 18ca 128c 99ab 1014 1876 19c0 9646 | 1837 107c 19fd 9a87 1be7 9043 9362 17c6 14f6 1066 1888 8f7c 9be5 9146 8fd2 15f4 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 15 | 146b 9a8d 1888 1ac6 1bfb 1347 9527 99e1 16b4 1560 1724 142a 9b3b 160a 98ef 97de | 1c8f 95be 1781 96ea 95df 9da8 1a67 9688 1af1 19f4 9b1b 1c0d 965d 9c5b 169e 16f1 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 16 | 9a52 9a9f 1282 11e8 96fb 10d9 12c2 91d6 1b4b 9c1e 919b 967e 15a5 9a4b 15c2 1294 | 92b4 91b1 98ad 9d0a 9506 9ae5 1c3f 1c63 918e 11a4 1d54 1707 1853 9201 961a 1975 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 17 | 8ea1 958a 8e04 91eb 0fa7 1063 9904 91b2 0fc2 9181 8f42 0edb 1558 13a4 128e 168e | 1ab9 14ca 998c 19ba 195c 1a71 0fb3 96d8 1962 18bd 9ba7 99f9 92c2 151e 9783 1414 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 18 | 9126 8fe0 99f5 167a 1552 17a0 9728 9a46 131a 97cd 92af 158a 1a6c 9320 1340 13c7 | 9b5b 1bec 9364 15b6 9931 940d 1853 1190 1a61 9633 9b7b 987b 146f 988e 1aaf 19b8 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 20 | 1bca 1916 1389 1545 98f7 174b 12d4 1afe 1ad9 141e 194c 97f3 92a6 9183 9af1 9477 | 9579 17f0 9cda 1b65 96c8 9987 9ca0 97e3 93cf 1a81 1715 96ae 1f25 9cfa 16cf 9bb2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 21 | 1e76 1b7f 9336 1a5c 991a 1421 9942 96c1 9490 1b7c 9e0c 1b11 92f5 1331 950d 16da | 9350 13ae 9c0d 960b 1971 1a50 96af 1abd 1c9e 931f 1404 9541 9cc3 9c44 1a23 19a1 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 24 | 96d5 13b5 929c 99d3 9b2c 8ee1 9c22 95da 9ba2 9bcf 953b 14f2 111e 198f 9188 152c | 1448 186f 9abc 929b 1294 9b2d 1146 9482 11a2 90c6 9857 9972 9abc 93ec 1c04 9676 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 25 | 1199 119f 986b 9103 197d 1433 1454 9b2f 1856 9430 187d 936a 1849 904a 167b 926b | 1b53 9ad5 9422 1a88 8fec 184d 1558 907f 93d0 1655 1348 191b 114c 1cfd 963d 99a3 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 26 | 96dd 9502 1ade 1698 1b34 198f 9bf6 1c5e 1ac2 13d2 1347 15c7 9749 9195 0fd4 9131 | 9728 14e3 927c 9775 91bb 9093 11a6 11ca 9019 9892 9974 1669 93cf 9af7 9ae7 9915 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 27 | 9bb1 1a9a 9d03 1d56 17e1 1784 929e 15c0 15a4 19b7 94e9 136e 152f 1b58 19f4 9319 | 12f0 1396 93a3 9241 980c 16b6 9cb4 1909 1996 98f6 9a53 9c94 1bdf 96c1 964a 1e49 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 31 | 10ce 1a6e 9c21 95f9 14bf 1263 9cb9 9b73 10d1 1c66 1276 1ac3 9531 936e 94a2 9191 | 9d72 9423 9205 99ee 9bc4 1cc8 9276 93af 1c5e 927f 9e59 1335 1979 1a2e 1911 1c20 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 33 | 9426 9c2a 18cf 148e 9d71 94ea 9811 15c8 1401 19ee 9cb4 155c 9cfa 9863 9b28 9ad6 | 9cb9 9525 1507 9b6b 943e 9973 19cf 1a41 9c45 963f 13c1 9b8b 1508 9874 148c 1598 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 37 | 1a3a 16e1 98a4 939e 1312 13f7 1c59 1122 1810 959c 9b1e 13b8 1abf 19fd 9934 1a6b | 12d5 9688 1387 1bae 1a56 9c20 9159 1dd4 961a 994e 1483 1a75 13ad 936b 12bb 14be | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 38 | 1ac1 9689 1999 19d5 9b2e 18d5 1526 18f1 185b 968e 95ab 97f4 92ca 9004 1703 934a | 94eb 1517 94e6 9303 9477 93d7 9986 9467 9681 963e 958c 1572 1b6b 1c94 151a 9991 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 40 | 1877 13fc 153c 1520 163c 1ac0 9496 981e 9165 1c3e 983c 1ade 9d0a 1d47 9848 93f9 | 175c 9ad6 1a67 9af3 9bb5 960f 9cc4 997c 1df8 1352 17dd 9371 1231 93da 9781 1b98 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 41 | 93c9 9338 90b9 118f 95eb 195a 192b 1718 9980 9281 15a1 1152 0fb5 1541 187f 10eb | 1743 1897 18ac 19a0 1582 100e 0fbc 12ab 90ac 17fd 1484 9b50 9b15 95c5 93e7 1adb | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 42 | 9789 1776 1229 996f 1957 19f3 97e2 1329 969e 1ab4 9aec 169f 1393 1636 0f93 9650 | 12dc 1557 19a1 11e7 915c 1297 934c 9ad0 1599 9007 9133 138e 18bc 96a9 1ba4 1506 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 46 | 9630 956c 1288 11b6 9b54 9668 91ca 9922 1aaa 99da 914a 1b4c 176b 9c65 198d 995b | 9555 1635 1aa5 99dd 92f5 16a1 9a2a 12d1 9106 1317 1a2b 9324 1715 1219 93c5 93d0 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 47 | 163a 98f4 10ac 9337 1357 1907 9b6f 15e6 15e3 12a6 94c4 13af 17b6 19b9 9805 97b4 | 170e 95b8 1d5e 98ec 9913 154b 1012 9904 9769 9cff 9993 9bc4 976d 93d7 957a 14e8 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 48 | 18a2 107c 15a0 967c 97ea 119d 9b65 958c 15fa 1571 99b5 140f 1b42 102b 93e7 1854 | 1483 9b81 18d4 18b5 16f3 1a68 1188 956b 95af 9695 141b 9778 9072 1b08 9936 95bb | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 49 | 99c1 1a31 1483 914e 19c3 15c6 9673 9506 9788 9293 1372 919c 8f42 943f 9b15 97ea | 112a 13f0 9917 1c60 11f1 9667 952a 174a 1347 1b2b 9aa1 9a56 1be3 96bb 9263 13d5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 53 | 9d5b 91f5 9c43 154f 1aa4 927d 95d0 1347 95fd 9a27 1783 1307 1526 1ae7 122c 1d23 | 1530 9e11 95ec 9c14 96e7 9c93 9a16 9d97 9c36 17ad 1852 1bdd 9cdb 9670 9d68 9497 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 54 | 154d 998b 955b 9252 91b3 13e6 0e93 959c 9436 92e0 946d 1664 1880 105a 9700 9ae2 | 976e 1123 16ed 991e 172a 1552 1ae4 1763 183d 9624 965f 129d 134f 9ba8 9531 106b | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 55 | 95e3 94bc 172e 184d 973e 1401 9121 1942 9924 1633 9510 932f 9926 9130 186f 18a8 | 1745 197e 14a9 9672 163d 9a1d 9c74 9611 93fd 18e2 19ab 1924 93f9 1a79 94d2 9558 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 57 | 952a 1935 1390 8e98 9645 18df 914d 17bb 8fd6 8fa6 98b0 97af 0edc 9764 14ef 11bc | 15af 0fb1 1857 1a4f 1641 10df 1959 1288 997c 9996 1318 11a1 9b4f 1189 9857 98d2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 58 | 144c 9a4d 0f2f 942a 19e3 9292 935d 8e8d 9491 97fa 8f6b 120f 1312 13e5 9734 0f51 | 16d8 8f29 9b66 16ca 0f9c 17eb 9774 9a94 9460 8fca 99af 1802 9791 951e 106f 9a07 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 59 | 1804 8e4c 15f8 11b5 93b1 158e 1667 8fe8 900c 9979 9203 99be 9679 9638 1818 165b | 92fa 1a93 9568 18f5 15e1 15a4 9237 9871 1bf8 104d 1677 0eeb 128e 92c4 1358 933d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 62 | 9739 174a 9ade 94c2 19fb 9432 1afa 92e7 9787 92dc 124e 8f44 95a9 168d 19b8 90a0 | 1472 91a0 10f7 93cc 8ec4 96a0 0f81 1967 12d5 195b 190c 9a30 12fc 93f8 91e2 19c2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 63 | 13dd 979a 987b 181d 1a79 1733 9a8d 91a9 134d 95ff 14fa 11ab 11de 964a 9670 1cf7 | 1972 9657 178d 168b 13aa 1547 95a6 1dbe 1b5d 98f6 1aeb 9ad4 9d97 16b0 17c1 9307 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 64 | 14ec 93e3 96e3 127e 1329 17cc 1250 97e8 99ed 1885 1b0f 17eb 17fa 176f 9baf 19ad | 184e 1ae6 1716 9a8f 9a71 1367 98bf 1320 1386 91b4 920d 945c 156a 1778 91a9 10cc | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 65 | 1958 93b1 1466 93f9 115d 1420 166b 1465 119e 98ad 10b3 9408 9676 916a 15f7 96d1 | 9065 977e 15d6 94bf 1826 9761 9434 97db 96ce 8fcf 9b7a 95fe 9285 1aea 15f6 91e3 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 66 | 999a 9695 9b94 1867 173f 935f 9b35 9ad8 1421 0fb8 950a 9b21 91f3 9938 1424 9010 | 109e 9445 91ca 9456 14fb 97a5 10b9 8fca 1880 9c2f 965e 91a1 1c5b 1324 98ba 9b0b | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 67 | 148a 9cfc 1efc 1921 18b3 1611 9c07 1516 1670 1bb5 9d55 1e27 9664 1cd5 1970 1dd5 | 1c1c 166e 93b7 1906 96e2 9c8c 1454 9b7c 9ca5 959e 95fd 947c 9aee 92e7 96d5 1476 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 68 | 12db 149a 9b24 181d 9965 1341 1319 9200 14c5 160a 1b8d 91ce 9212 179d 13bb 93e6 | 188a 970b 909d 12a7 14f7 19fd 19f1 1b59 1590 170d 921d 9a58 1872 9375 1ac9 1a11 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 69 | 9a73 96fc 9139 92f7 1231 1618 19bb 1011 166f 9abc 97fa 12ee 9193 12c9 19f0 15ee | 8f4e 14e0 986d 96f7 18c7 9707 90ca 19f6 9623 1062 1293 971d 1916 1ab3 12a4 97ad | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 70 | 123e 958a 161b 994c 96b2 91cb 979a 0fb5 92ff 93b3 99fd 1846 8f99 976b 9872 183e | 16b7 95b1 9159 124a 12d1 1910 92a1 99a2 1717 1693 90f5 8f6e 183b 9207 8f57 9224 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 75 | 10f4 1230 1aba 95f5 1767 193e 1704 15e7 9a60 9711 1566 91b7 9b5a 1534 1a54 8ea2 | 1b94 1b24 9214 9679 168a 926e 9497 9877 901e 9449 949c 1a80 10dc 96a5 1206 1b1d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 76 | 1150 968e 13ed 143a 94ea 18e5 99f6 9a76 1969 1b2d 1329 9add 17fb 96ae 1233 9539 | 98b5 9674 9672 1596 174e 9125 9208 90b4 10af 90d4 99e9 1349 921f 1502 1b3c 94ca | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 78 | 9372 0fcf 9963 18b5 8fa8 161e 9260 9936 9144 90a5 1209 9322 9391 8fe1 9066 98cf | 9679 99cf 124c 110b 1bc6 1445 98b0 10a0 1703 992f 9725 18d2 9780 98c0 1bd5 9182 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 79 | 1b5e 9270 98e7 9ef6 16ad 9be3 1bce 9e36 9bb5 9acc 9da6 936c 1c61 1843 9e82 9c16 | 1644 1e2d 9776 9396 1b02 96b2 148b 9365 95d5 157a 1591 9d69 9728 1a82 947a 93e9 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 80 | 95bf 1636 1758 1472 127c 1c6d 1ae1 97b2 179d 924c 159f 1a2b 1a83 914c 9bb7 97f0 | 1808 174b 974a 18c8 9df2 93ac 1271 95d5 169f 9dac 9781 95e1 94be 9c75 118c 9533 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 81 | 9363 14b9 9922 9b48 10ca 1990 92f0 931b 9839 1481 1cec 9841 9581 11c9 9c3c 1100 | 9d8b 1a49 9708 156a 1cda 172e 1bcc 1c33 9897 1af5 946a 9734 1b74 1d4e 1443 9d00 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 82 | 934d 1598 1b72 977f 1a4a 1365 19d1 9b2b 1cc5 9c8d 97bd 16a6 983a 1354 1574 958a | 1c7d 996a 9310 1642 1283 9b47 1506 1282 9257 1289 1485 1956 9556 9a47 976e 9991 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 83 | 16ab 9388 9a07 9683 1a27 991a 985d 15b3 156e 1b14 926c 9503 98fe 9537 17c8 135d | 1a59 9ca0 9390 1a71 13fc 97f9 1897 1916 9c37 1512 1b35 1c11 1591 19fd 196b 9b0f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 84 | 91fc 915b 96d8 1ade 128a 17ad 1848 1961 964c 16bd 1452 9830 98be 14ec 91eb 1399 | 1cc2 1be1 99e3 965b 9ab3 961d 94bf 14ee 1a1c 18f8 9c87 97e1 9710 1a4d 1c3b 1c54 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 85 | 9426 9e0a 1944 18cc 1275 9e3c 15fe 1983 94e8 1ad5 1344 12aa 9593 9de9 1bcf 13dd | 1e80 93d5 96cb 988e 1d81 1481 19c9 1819 1b71 14ca 9d8d 9dd8 1ba6 159c 97e0 1f91 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 88 | 1540 16db 921b 8e8d 97e9 98d7 91a8 151a 1903 9549 8f68 9062 90e4 95d2 90d9 0e20 | 15c3 1321 9890 9960 9471 1283 178d 94ad 92ff 9325 9a33 1918 9a46 9290 1b70 19b1 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 90 | 1ba1 1c06 99f9 921f 9b54 933b 9a2a 9405 11a9 9d71 96c1 13f9 1ae7 970e 9315 96c8 | 12db 1194 14fb 1bd0 9389 1ac9 92fd 9866 1b4d 131f 983e 98ca 11ff 98c2 1d30 159a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 92 | 9a3c 19e3 1643 9a4d 166d 125c 15e5 94b4 975c 1890 9c1d 1c70 96e9 1b33 1a55 9a37 | 1652 9547 1888 1516 1a47 9bf2 9a22 18dc 989d 14eb 130f 11a6 9937 95c6 14ed 9602 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 95 | 9b4e 96c2 1e1e 9ccc 1505 17d9 921f 9bb6 1e63 1558 975f 149b 1970 1754 1609 9bfc | 96d5 1c07 1500 1378 1ab6 1848 9e80 1617 1375 9d78 1add 1dc9 9803 1a5a 9d4f 9645 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 96 | 1721 96fb 17d5 9a2c 9201 1a5f 19ff 9af4 19a2 1b88 931f 1264 9e61 1d7f 1758 1c7c | 9845 9a09 193f 95cf 1b68 1319 13ab 16ea 97f9 952f 9cac 1db6 137c 1246 1937 1117 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 97 | 1c15 99e5 9e40 19b6 9975 1c9a 9cbc 9523 983b 1413 9cca 140b 17ae 963f 9875 96e9 | 96f2 9779 1481 991b 9912 93b3 1387 1e3d 17b1 9d7e 145e 1ee4 9aa2 1c37 1a4d 195a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 98 | 9338 1d81 1cc1 955e 9c91 1d47 9396 16bd 14c6 14ea 143d 9819 17c9 9404 9b91 1aba | 1cf9 129b 954d 9d2f 939f 9338 1ff4 9b4c 9bc7 1bc5 1cb2 9b0f 99fc 1e1d 142f 1719 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 99 | 95ff 157f 1930 93e3 95a8 1a9e 153e 1391 9a7e 1448 987d 970c 9c56 9939 9627 9b2b | 1b5e 9bc9 165c 9c66 18e4 9643 9990 9da4 14e4 9b32 9602 98cb 91a1 97e2 9b65 162d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 101 | 1018 189a 1566 1335 1734 18b3 9abb 94f2 19c6 1931 12fb 9999 912e 1be7 1819 943c | 9b74 93ff 9712 97fe 92a5 9115 1371 977a 106d 1324 9b2d 917b 1b1c 1165 15b2 19c9 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 104 | 1016 1987 129c 1233 1728 158a 145b 9bfe 9693 9a78 9412 94dd 18e2 1a3e 9759 9594 | 9b0a 93c3 9ca0 99f6 95b4 98ab 9744 9246 9834 152e 99e2 9779 15b7 1175 15c0 1655 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 106 | 1cec 167f 19f0 9245 97d8 1bec 17d1 15a4 938d 94ae 1761 9581 128a 18ae 1720 956f | 93ca 9a5c 1782 1da7 9b6c 1497 17d1 9d16 1de8 9c73 1801 1c35 1dba 186b 9b82 1d98 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 111 | 165b 98d2 17e7 15c2 1283 9801 91fb 9575 104d 99fe 1990 1451 9147 96a3 9117 1820 | 1311 8e90 108e 12ba 1726 0f2c 9954 1629 18be 1020 90af 972e 19e5 9387 97cc 1032 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 112 | 12e5 9229 9757 1139 1968 9b16 19c0 1243 9795 1cb1 12f3 0f97 92e3 1ad5 9a4f 90bd | 19dc 9baa 1427 1b46 1327 9125 1397 9c55 153e 11be 98a7 1bd4 9a68 94d5 135a 9b03 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 115 | 99d7 1c39 9780 1c38 939c 1920 9478 1a6d 1cb3 9572 1a34 19fb 1bc6 93d2 1a2e 992b | 954e 1402 14ab 9393 9bd8 1618 1c17 1651 12ad 1b57 9420 12b6 147d 19d4 966b 9628 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 117 | 9570 13d3 1a08 95e4 9154 1a2e 1a6f 121e 9108 13cc 14a5 9257 9121 1b0b 1a29 97a1 | 1a64 1b48 136d 9867 9b59 9389 9218 9b61 1dc8 1a55 1a0a 9cee 1d41 92c8 1256 15f7 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 118 | 954a 94d0 1d56 1d86 1936 9b9e 1da3 1689 1757 1271 9df3 1b1e 1da0 183d 961b 9c20 | 19aa 9de5 9462 940e 1742 93f4 928e 987a 1890 1d66 945f 9794 146b 16a2 1c54 92a3 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 119 | 91a8 8ffa 1298 923e 16a1 9431 9903 131c 1a97 9617 9241 1ae2 10bc 989c 928f 95b5 | 9a8f 9adc 179f 1888 1383 182d 11f8 99aa 109b 9619 1819 114d 1ae7 1433 9973 968b | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 123 | 1290 178e 99fa 1833 11da 1372 98f1 9afe 929d 927d 167e 1459 9d24 919b 932f 90a8 | 1abc 9573 962a 95b8 9b1b 9c19 94b5 934e 1dca 9c14 989f 1a0a 9208 1e4c 1bfa 9bc8 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 124 | 91fa 985b 16c0 9b5a 9488 1b1e 93e5 180c 159a 98bd 13f0 91ea 1b2a 93fd 9942 9b9b | 1a74 152a 94b6 90e5 97c1 915b 9911 15fd 17c1 93f6 1799 9bcc 93b8 19af 92b3 1370 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 125 | 185c 91e4 1200 9be8 129d 9a2a 122a 98b9 152d 1c29 13d6 1591 9900 99f7 95d9 9aab | 17b3 1ba7 9ddc 14b7 9a94 94af 1b36 1796 9a18 13a7 9ac2 17ca 166f 149e 9829 14e2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 127 | 991b 9aef 91c6 1b18 10ff 1621 16db 9464 9835 94cf 1920 1126 916e 979f 983e 1a3c | 124c 1239 9c5c 90cd 9c9b 17f7 1410 1999 1496 97bd 12de 9b79 1ca0 922a 1216 928f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 128 | 10d0 1715 9a1f 96ec 8fa1 96f8 992c 95d3 917b 96fb 99d9 15ce 9928 8fb4 9595 90fc | 98e2 110d 907c 12fb 1aa7 9484 1231 94fd 1b47 1348 8eed 1627 1305 9bd1 953f 99b6 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 132 | 90a0 9041 8f16 1352 9941 9918 9a2d 1412 13d4 1b01 9253 15de 9018 1263 1613 97b7 | 1be9 1b4e 1aab 96ff 9189 1459 8fc7 992b 180f 9278 9acd 14f7 9cb6 9898 157f 9597 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 133 | 1b22 1aa3 9542 19c9 1162 9487 1c40 9307 196c 1aed 9b29 91b9 91e1 958f 18c1 9826 | 11de 9473 1a55 936f 9bd5 1828 1308 1a39 94d6 12b1 14dd 1c27 1bc3 9a81 1460 163f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 136 | 939b 1d65 9b47 96c4 1b25 9520 19b0 975d 96f1 18bb 114e 12e3 1a8f 9965 9a67 1a4d | 9a12 9407 9405 19ec 13a3 98cf 14ab 169b 195c 16d1 1d45 1d91 13d9 15c6 16e5 9538 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 138 | 1464 1c3b 9501 96ff 142e 1988 1185 1260 1b26 95e7 9c78 1a2c 1633 9dc0 1d20 96e7 | 1cf6 114b 9ac0 9aaa 9cbd 15a7 1cef 9dde 1488 1b2c 1167 1325 190c 1323 946c 99c5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 140 | 9d12 96ee 1c47 99e1 12d7 9453 1564 150e 989e 9345 189e 179f 18ae 9b08 92bb 1bc8 | 130e 1891 9386 1477 9cf0 9a88 17d6 19bd 961b 1a92 942f 98ec 165c 9311 9c6d 11b2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 141 | 13ca 976b 996c 9484 13a7 95be 1761 9cce 1c2f 9b51 13bc 1d9c 1978 1408 913f 9ae5 | 1a44 1760 16bf 9c91 9df6 9a37 99fc 13df 942d 133d 1d26 934e 9799 9a86 9ceb 95fa | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 142 | 1652 9ac6 1be3 1b01 1093 1b0d 9120 1a15 19e9 19d8 9657 9278 9180 12b1 1948 989c | 97e1 12bb 90d7 932b 9cf2 1380 1c55 91e5 91c7 147a 9633 188e 1968 187f 1417 9203 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 146 | 1027 107f 901c 1466 9355 9b43 92a4 1812 12ce 1675 16dc 17d2 1286 9156 19fb 1602 | 98ba 9bd1 9af9 95d0 9932 9171 195b 133c 1905 96e3 14f0 13bb 9957 1b5c 906e 9410 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 149 | 1756 18fa 1289 9a25 9866 17c2 9731 989d 93ea 1413 978e 95d8 13f9 1188 995f 1028 | 96fa 9663 9991 1243 1743 96a9 1766 9593 98ba 1792 183f 98c0 19d1 1c07 160c 1c78 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 151 | 187e 914c 13a2 1178 9452 9762 1721 9a0b 1384 98cd 9620 1aea 9ab6 1c2f 922d 12f4 | 15e7 9d3e 1b66 9c2c 9c3c 182f 1634 16ff 1b60 96d8 99b9 96c5 153b 1140 1ded 9b67 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 152 | 1304 1813 143f 93b3 93fe 9262 18c3 9a4b 9473 166f 170a 95a1 9554 912a 1529 1438 | 98a3 14e4 992c 15ad 199b 9aae 91a7 116b 959a 94fd 95a0 1390 960b 1905 9865 9607 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 153 | 9dc0 1c6f 9b14 96ff 1c74 1dea 9916 9b9f 13a4 9326 1e2a 17b2 1495 1446 9470 1ac1 | 93a6 9526 962e 19b4 1356 93fb 16d1 9224 1c14 1d0a 1371 19ca 1bd3 1bff 1d45 1529 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 154 | 13a9 14b0 1889 96ff 980b 117b 9105 8e98 1a45 91d2 148c 10af 114c 973f 16a9 9a0c | 9638 12b1 10cb 9321 112b 18c0 16aa 19c6 0f8a 95f2 92cc 9a79 1692 910b 944c 8fb2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 157 | 9586 1194 1470 985f 97f1 9af1 1be2 1561 918a 9193 932a 1655 16a8 173a 1682 99a0 | 9865 1877 9892 1543 9454 132c 90b9 1628 1813 9b20 18ad 142f 168d 13d2 93e1 10f9 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 158 | 1151 90b6 1490 1d6c 99de 9083 1599 929e 95b1 180a 95f9 94ef 169e 112e 1af7 9aca | 9ca5 1caa 98cd 9261 166a 1d5d 1973 19a3 1734 966a 97cd 9850 162f 1cb7 9100 93c9 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 159 | 1615 159b 992d 943f 991e 1a23 1307 9b86 9268 136e 142c 1b8c 9701 90c6 9471 0ebb | 15ab 96d6 11ba 1898 11c4 124c 999d 10d3 197b 9902 995e 91ec 1448 1ce2 165f 9b3f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 160 | 109d 9524 148c 1795 9427 9485 9489 9a67 9ab6 18e2 1693 19a5 9282 9a50 9ac2 9b2e | 1cc3 9703 98c2 133d 19f6 97d3 172d 91c9 90f8 919c 9648 9125 9b63 10b7 0fa6 1204 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 162 | 14cd 950c 1a45 1b3a 1703 1221 1972 1837 1d36 1a6d 136e 1750 9957 95d5 99d4 96d9 | 1b86 1d94 96de 1685 9b8e 1d9d 9763 997a 1319 161e 9cfe 9b21 17e7 1c3c 9556 1904 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 166 | 9592 1abc 93dc 19c8 1a7b 914e 9837 16b5 157b 985e 1a06 955e 93f9 9405 9913 9593 | 12d1 0f85 17e5 1138 917b 98bb 903a 9531 149f 13a1 1142 15ff 9407 970d 926f 94c2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 167 | 1200 1410 9577 166e 9580 988c 1dab 1806 9b0c 976b 136c 991e 144a 9339 1bc6 17a1 | 9e9d 1c36 1d8b 9a54 1c60 9908 1360 16da 93cb 1b2b 9d44 9931 9b64 9e3d 13fc 17bf | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 170 | 9773 1d4f 113b 17a5 9122 98aa 17d4 924a 17e8 14b0 1510 1982 1b1f 9ad6 9aa6 1275 | 976e 9229 9c17 95ff 1b3e 14a2 165d 1ca9 95aa 9a69 9860 928c 1331 13ea 10a9 9c30 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 173 | 994a 995e 1c19 9bcb 935e 9c31 1bc0 1330 9317 1ba0 1296 11f9 9cd4 92b9 9c7f 9583 | 1402 173b 9410 9287 9d23 94cb 937d 9a1a 9c16 9332 1de9 9b93 13e0 1b27 148f 980f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 174 | 12a9 99de 14a2 9149 17fc 8f2c 96de 1053 911c 126c 1296 18b8 9b56 103a 9269 16c5 | 99b6 91cb 96f3 1889 93f1 9a21 13fb 9cd1 98b7 17aa 1944 90b2 9175 9a19 98ce 12a7 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 175 | 973c 998f 1425 92c7 9930 11c9 1887 92c2 9d3b 970c 955f 995a 99a9 188c 9724 99a5 | 1833 180b 1be3 9dae 196c 1d80 98f3 9db0 1247 199e 1c5a 1723 976c 968d 9a7a 162b | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 178 | 96a4 1427 1487 1904 148c 14a2 1b05 1944 98d5 9372 9c6a 9a43 12b5 1ac4 920e 19b4 | 99f4 9c9d 9d6a 180f 9ba1 9c8f 1476 148c 9606 9ca7 15e8 977b 9f44 96e7 1c44 9620 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 181 | 9329 19f8 9819 9430 10c4 1573 9129 1547 95b8 1864 94f1 16f6 91dc 1644 1a03 9053 | 196f 9322 1270 18aa 9c96 97b4 9a1e 9895 160a 1261 16f4 163d 9c4a 1668 93b1 1a5a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 183 | 151d 9047 1ac8 990c 95b5 1711 953d 187c 0e7b 14a4 1678 1315 92a7 9922 9654 1073 | 13e9 18b1 10c6 0f9e 96f0 9566 95c6 118c 1a58 9770 922e 9572 97f7 8f65 157a 9c75 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 184 | 1260 91a5 9549 1750 91b1 16b5 9b1a 1470 9282 9ab5 95af 14d8 0ead 9209 1875 9ab6 | 17e1 1a0d 9599 949f 9979 9480 100b 9689 179b 0f98 95ca 9645 9984 1908 9284 1160 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 185 | 16ee 11eb 918b 911d 95a9 1b6f 1bb2 14a5 187b 1007 1aaf 92d6 9bcc 9ba0 196c 13a1 | 9513 9a08 9956 1bc3 9795 10eb 12ba 1a39 967d 9c9e 9141 1a70 9254 1248 94b3 9b21 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 186 | 19d1 9974 1816 9911 1d38 1734 15f6 9813 157e 14b7 9729 9397 92c0 9ad8 1907 99d7 | 1655 9460 15f8 145a 125e 9a71 9a4d 159b 9c56 19ee 1660 9c92 9b29 92b0 16d9 153d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 187 | 97e4 92ad 97b5 149b 119d 187a 9695 933c 155e 9420 1056 9158 946a 91eb 1488 8f20 | 92b8 177c 10a9 1634 97c2 909f 9164 94c6 9524 9678 99b9 9977 1456 9667 9661 1ba6 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 188 | 1cf2 1834 958f 9d00 990f 134e 130c 96cf 9a88 94ef 1443 17d5 94e6 990e 9454 9431 | 925b 96de 1945 923f 14eb 9b8d 1938 9652 93d1 9a74 9b86 9705 9720 9649 98af 1a19 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 190 | 14c0 962d 956e 1092 12b3 1295 125e 9267 9025 111c 8f29 97ec 15e7 17bf 1ad2 980a | 1551 17f3 971d 1bfa 9a01 985c 1a54 9b06 9c41 18fc 1bd1 13f7 9749 9519 10f4 93f1 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 191 | 9234 1932 1268 115b 0f39 9189 10b0 992c 1193 9638 927c 9368 8fb2 119c 16fd 8ffd | 96f9 9080 15e9 1792 9870 1777 98df 91a7 19a7 1545 1825 94d4 9873 9749 92f0 1a6c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 192 | 1165 92b0 11ed 915f 188f 10c9 9880 959c 955a 1417 1602 93c6 1772 16c6 9964 95ae | 18f3 1b0b 9880 18e3 13f5 9aca 948f 1529 9646 96d5 1641 99e4 95cd 9637 1499 150e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 195 | 11f3 9568 1d0a 1934 1679 1152 9b40 909d 96c9 112f 9622 142c 9b61 94b9 9bdb 99a2 | 9d1d 19ee 921e 1677 9922 9df2 9198 1bd7 16ba 9bd2 150a 989d 92c2 9a3d 91e1 130c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 197 | 9aba 1b21 9d34 128d 9464 99c4 1770 933b 9adb 9ab1 96e1 964e 1cab 92c3 1121 188f | 1559 96c6 1367 9c57 1d3a 1771 1adb 1b4f 963e 1625 1aca 1843 12f8 1e03 9d5a 16eb | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 198 | 9b31 936a 13d2 9537 12da 1a43 1716 1973 18d7 1012 9bbd 1847 16ca 18f3 9912 9088 | 124e 9ab6 9835 95dd 9c6d 90b0 1583 1215 92f0 1c43 92dc 13f7 15f7 9547 15e3 1ceb | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 199 | 93cf 1121 9192 104b 106e 1434 9207 9a02 9881 947f 0fba 190d 0e9a 9094 11df 14d0 | 1589 17b2 1831 19d3 1983 968d 19e1 9020 103e 15a7 1bc1 8fe0 9a15 1891 9990 153a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 200 | 1407 1728 9738 12bb 96d0 147e 98b2 9690 91ad 149b 10a8 1192 97e0 1324 0f4e 1ad1 | 98b0 14f8 1476 9a61 938b 98dc 943a 941a 9b5c 9644 1b2b 193d 153d 15a8 9b03 9226 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 204 | 9ac9 9923 963a 9507 118d 958d 99ca 9c0f 9bd9 18fe 1405 95fd 975d 9a4f 1bda 96dc | 910b 154f 9896 96a1 9d7a 1901 934b 1347 12cd 133d 172f 97e2 9445 1492 90b0 9827 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 206 | 1353 1302 1af9 1335 1c35 15d6 1903 1cf8 15bc 94a5 9d67 9985 1840 9bdf 9366 9c35 | 9c98 9d68 1438 9e12 9575 9b17 17e5 1223 19fb 1b92 12ac 97a7 16cd 1730 9f3d 9321 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 207 | 1262 9663 1ab2 1a3d 1a16 1907 1447 97ba 1770 97cc 943c 14f8 9600 1374 9cf4 1a09 | 9ec5 16f8 13e5 13c7 1401 96ae 9c95 1630 9604 986f 1aec 1acf 9abf 9c9d 91b1 135a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 209 | 172c 15e2 13ef 9a49 92f5 1706 1593 17ac 9427 19dd 176c 8fc7 193e 99e6 11ff 161f | 15fb 974f 987c 9004 1af4 13bf 94d6 9294 1585 105d 1642 9c01 9420 93da 99cf 96c5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 210 | 1d47 1f6d 13b1 9bb2 9775 15d2 92ba 1e1d 149b 1df6 9be0 969e 9584 1425 1709 9359 | 12dd 942a 1d38 9395 18b7 9a3a 9d4d 13bd 9efc 154d 95ee 1b1f 9b70 1c66 9965 9e3e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 213 | 1ab0 1344 1988 95bc 9545 9ab8 1965 940c 15de 15d3 91ab 1959 950d 139e 19a8 9617 | 926c 181e 954f 15a7 9915 12fc 9407 17b0 979c 1534 1bcd 934c 181c 99ef 94d4 18b3 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 216 | 9138 104e 1497 972b 1436 926d 19cd 17df 93c5 96eb 1386 1a5a 961d 1c20 9236 1131 | 9ba6 1ae7 9922 9447 1ae9 9937 942c 9748 9a6e 96b5 9b1f 11cc 963d 11fd 1ccc 1ae7 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 219 | 9a92 1548 1903 1512 151c 1223 933d 1778 1035 1740 99bf 96c4 932e 12fb 17fd 1103 | 1056 15ab 12ce 162f 142d 9867 95f5 1288 9bfd 9351 8f10 13ee 1680 9717 12e7 97e6 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 220 | 9bd1 93aa 9bd9 1352 96a9 9512 14f4 986e 93d9 1148 9996 97d8 1874 98bd 151f 115e | 109b 974c 108e 9632 133f 96ed 1706 1558 9620 99dd 13c6 12b8 9386 9183 16c1 9907 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 222 | 9a3f 94b6 9c21 9176 1a7d 926c 99b2 158f 177b 91f6 9a71 1abf 112e 9869 9a29 18ec | 1674 1bf0 1410 9c04 944f 9e55 1310 196e 973b 9ced 922b 92cb 1d47 15fc 93e8 97eb | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 223 | 1552 8fb2 9724 19dc 15d9 8fdf 9914 951b 9002 1486 1a04 1781 9129 1628 9ab1 9634 | 9734 19e7 942a 92f7 1534 9b6d 1192 96f6 1cca 9606 1099 9208 1c15 13e9 926b 146e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 224 | 9aa7 19ba 1b0f 9602 1b30 1060 1728 11b9 9a11 92b1 9b41 16cd 14ff 1356 9b59 90e6 | 9143 1280 13b0 16e4 9479 9bba 982d 1c3a 9168 9a50 9114 14db 9777 9af3 9322 9d14 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 226 | 128a 16da 1935 17e1 9cbb 1ab7 1708 19dc 97ae 97cc 9857 9c48 9899 96c6 1810 9931 | 1cae 178a 1501 188e 121d 95ea 165b 17c2 1790 16e3 1532 12f8 9693 1af6 1567 16e5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 229 | 9151 921f 99be 8f81 9451 97f2 194f 95ac 946d 91c0 98e3 9586 95cd 910a 1279 1131 | 9a2f 1944 12b2 9b19 1842 91dd 937d 1309 97ef 997c 1059 9431 145b 1b46 1806 18c7 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 231 | 92c6 0f16 138d 1674 187d 0ff0 19a6 9040 1851 0f46 9651 8fa2 9205 9204 0fae 17d6 | 1a3e 1a59 96bb 16df 9520 9aea 1360 194e 1193 1a0a 175f 9adc 9a98 9a24 9bd9 151d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 232 | 95a2 18c7 1557 90c4 1a59 11ef 18ed 1998 12b0 1a52 9465 1915 8f38 9625 94b7 116f | 1679 1359 9819 9aa1 11ad 9bcd 94a7 8f94 9810 1168 9846 127f 1ba2 165c 9507 99e8 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 233 | 9173 10d5 992d 12d8 11fc 9569 199e 1853 9938 1ac9 9840 1a33 9a41 9697 1619 1b36 | 9a43 19d4 11ee 9ade 9999 18da 913f 9480 91b6 936f 9443 142c 143c 9399 14fb 10ce | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 234 | 14ef 9a3e 1354 1bb6 13ec 9134 1a67 9799 99f4 9976 183c 1904 1a50 122e 9d7f 1789 | 9932 1603 1bfd 92b9 9c90 9c6a 16ed 95d9 14e1 1769 9637 144e 94f7 1c0c 147b 163b | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 237 | 1243 1000 1957 9b93 99a4 1283 1142 17e7 1817 1580 159d 1126 1975 12e5 9ac0 9226 | 9985 9983 1270 109e 11c8 1939 9c8a 9605 145f 9550 16c3 1c49 9330 9b69 1238 1950 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 238 | 1729 11b6 90ab 1453 1a48 1469 9711 1ad7 99af 188d 110c 9ac5 8edb 17ef 1ba5 154e | 955d 97a7 9845 9665 91bd 954c 129c 1080 1301 1188 9ad8 917c 1a0c 952a 9145 963e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 241 | 99f9 1abd 18a5 13d5 92d2 949b 94f7 1ae2 9a7b 9b0a 1b35 940d 191b 1140 1a0e 16b8 | 9472 95ff 159e 9a84 1eb3 1c30 1cec 16c7 945b 960c 954f 1b57 9734 9dcb 9688 97e5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 242 | 1715 166f 9116 8fe8 0edd 1278 8f83 9015 14de 9062 191e 95e6 1199 19d5 1471 13dc | 130d 134c 980f 97bd 1a84 16a5 1909 9942 96c1 1b12 902c 151a 961b 0f5e 953a 152f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 243 | 132c 0f95 18c3 934e 8fd8 9447 139c 1816 906b 94eb 918e 90c0 1852 0f91 1951 1129 | 1463 1803 115a 94fb 99ba 1445 1662 117b 99db 14b4 15ab 19e8 1225 98a5 918c 9a05 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 250 | 94a7 1249 92af 11ae 0f9c 92e4 9835 152a 9aa1 1426 12f4 174e 99a0 1839 1108 92dc | 168b 19c8 1792 1b65 1ba3 19b9 1265 950a 9054 1892 9a3e 130c 1282 9456 17a7 19bf | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 252 | 0fc4 0ed6 1123 1029 137f 943b 114a 1aaf 17bb 9047 9710 8e68 98ee 1104 945b 9134 | 9b2b 18df 99d7 183a 17b8 9304 17c5 0fe8 8fda 1afd 139d 1a73 1231 987d 143a 189d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 255 | 1eeb 961c 1218 9e33 1a31 97c0 12f9 9b77 1602 936f 970b 1727 1d15 113f 1bfa 1e59 | 944e 1aab 9e17 94be 94a7 1934 9c56 9485 1baf 9c6e 981c 18e8 927e 1dc9 1346 948d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 256 | 1734 981e 10a8 18cf 1415 9b56 9715 9317 186d 18f4 988d 9a0c 97c1 110d 173b 1a4f | 916b 13b3 9b41 90ae 17c1 108a 14cc 9720 12b5 8f83 1084 1121 9313 1961 142a 927c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 259 | 1bd1 1968 174a 1697 9ca2 1d45 1550 9a63 11b0 998d 1cfb 95d9 1598 991b 9709 953a | 1559 98da 18ab 1a9f 1448 9562 1a5f 1471 9c9e 950a 9378 18a4 1a64 97f7 9706 9ca8 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 262 | 13b7 16df 9b8d 1572 17cf 948c 19e1 9b20 1794 129b 9ba7 1a29 1366 9847 194e 97a4 | 9c8a 97e7 930d 1a48 9652 97ac 1434 94d7 9489 9bd6 1446 1580 1a2d 141c 1451 16f7 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 263 | 9a1b 1a70 1439 11f3 982b 93f0 91b0 94c7 9311 1a01 1a3b 149f 9bbb 1819 14b2 1ce4 | 1434 910e 9844 9b0a 95bb 1b24 1d66 98e5 192c 9468 12d4 172a 1354 96ae 16e6 1194 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 264 | 9306 95ef 1925 947a 1c00 9afe 18ac 9b78 9301 1837 161c 1119 940d 1996 9417 95d5 | 1ab7 9a3f 1581 1cbf 9519 1666 959e 1676 1ac3 1540 1925 1c74 9b20 952f 9c09 17da | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 270 | 959e 95b7 1764 1c25 954c 9375 95cd 96fb 1422 1967 9162 9910 9b27 914b 1c84 9d10 | 1ad0 991c 9a66 93bd 9a3d 1cc6 1b93 9866 9e3a 98eb 1de6 98ed 13d1 9dbd 15c1 9331 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 271 | 95d6 1774 9237 1c8f 9be7 14e2 90cf 11ff 1744 10ae 1171 972f 1a49 98bb 1597 95d6 | 18ce 95b5 9b7a 921a 11ec 9a5b 1d09 992b 150e 9bc0 1ddc 94f6 9393 1302 166c 9723 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 274 | 1634 9b10 19ff 9abd 95d2 9874 9a81 999f 9d77 98d1 1c6d 11cd 9575 150d 18aa 1a84 | 9ab6 1215 1565 139e 9923 97f2 15dc 1499 1243 9806 94fa 9bd3 98a6 992a 959b 944b | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 277 | 165e 12a3 10ae 9a4f 9016 1ad4 1a10 0fd9 17ea 1000 18a6 15d4 136e 994b 937e 1503 | 13a5 96c5 1a93 1188 1968 1274 1114 1af0 923b 9c33 9452 15c2 9941 104c 9644 9484 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 280 | 18a5 14fb 9469 16fd 1973 9ad5 9701 9af9 9d82 14a6 190f 14d1 9534 1ae1 1b4d 9b33 | 99c9 1d47 1b24 9ca3 16af 94e7 1a89 96dd 151e 9dbb 1709 1e80 1d2a 94b7 1814 9428 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 282 | 91a3 9329 8eb4 9909 1066 1688 9b11 9775 98e7 98c5 120c 92ee 16e7 9950 0fe2 9328 | 95fa 16f3 1b01 113a 19fa 9266 903b 1140 9155 1261 9595 97d5 91ff 127b 17f7 14d4 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 283 | 141a 98c9 19e5 919b 1d39 96ad 1946 99bc 98b4 9c05 9bf5 9b1d 16e3 1674 9385 9360 | 9ee2 1972 168f 1ddf 95a7 1975 998c 1653 997f 93dd 14e2 97ec 9a12 9aaa 9d35 1d8d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 285 | 9b14 9d2b 1488 9223 19c0 1174 93f3 13d4 147e 9976 1a37 1bf2 162b 1436 11a6 9ae0 | 1290 9214 9af8 9b99 96af 1cb6 1d1a 1c60 9b79 149b 946c 9254 9ad6 1a75 1ce8 9329 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 288 | 1acc 1111 1840 179a 9331 12a4 1a36 99ed 116e 1be9 1858 98a7 90df 9a74 9828 8f3d | 1197 1995 1442 15f0 192d 188b 915f 1089 1c41 9265 9484 9191 1b08 926b 1470 1ba7 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 289 | 0e56 92c6 18b6 997d 17b9 0e3f 98ee 949d 1102 9230 99b2 1b8a 129b 9b3f 184f 15a5 | 9afe 18f3 90d3 12e5 10f1 1a9e 11ac 94b2 9857 193d 91f7 10c0 98e0 9065 91fa 14c1 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 290 | 96cb 921e 9c9f 9d2a 1597 1358 18db 9b67 9b05 1633 189b 1545 18de 97e2 9649 147b | 1872 1b73 11f8 122d 9bef 1c94 176e 167a 95f9 9a9d 988d 1a76 953e 982d 9887 1c5f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 291 | 90d3 1aa4 92c1 9655 939d 8e9b 17fa 9127 992c 97ee 0fcc 15c2 1731 8f17 921d 126a | 9751 90ff 1919 94ed 9756 9ae9 942d 9a5d 1097 126e 1941 1461 12df 19fc 17fb 18d1 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 292 | 9790 94e9 17f4 9839 1598 989b 117e 9092 9899 1a9c 1a0a 9818 114d 115d 92d1 9110 | 1655 9666 14d1 12a9 970f 1244 9a1c 1bce 1481 913d 90fd 1546 19de 1cf2 9954 1d2a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 293 | 9c11 9344 0fd7 19f8 1897 13c9 1b25 9c2a 98a0 9b5b 92c7 987b 932c 9a77 194f 96b9 | 9256 9ac5 9c21 93f3 13f7 1bf4 9281 131a 939a 916d 1cab 13ad 99ca 147b 94e1 971a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 295 | 1608 18b7 13eb 13ca 962e 98ec 9af8 9793 94d4 91a4 195e 9531 9996 939b 93da 9bae | 96c4 9588 9953 9b0d 1608 93c7 14b6 1726 18fb 9bfb 943c 9787 12f2 9c0c 1a3a 119e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 296 | 993e 92a2 9bc6 9b7c 98e8 11e1 92d1 9520 9878 1915 953f 9614 1427 1cab 9291 9417 | 1707 9c62 929e 1269 155b 9e35 9bb6 1aeb 18cc 14ba 9a42 9b1f 9c02 13b6 9e12 9a2c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 298 | 9094 9945 16b9 9664 9a76 165e 0f9c 9524 18f1 147a 13ea 11b7 9974 9bc1 97a4 92af | 9a81 1449 16d7 96cd 9146 94db 9b08 1788 144d 9ab9 9af2 199c 1560 9197 93d8 1940 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 299 | 16b3 9323 99a3 96d0 179b 98cd 1591 9a11 1be3 17bb 960a 1b6e 9763 9a10 9761 951a | 9c06 9b8a 1951 9a0f 9872 979c 1b75 96f1 962a 995a 9b27 1457 1905 1635 19ea 9b43 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 300 | 11db 1958 1371 1c67 942b 9486 151b 93be 13cd 110a 91ec 156a 13ab 1930 120b 12a2 | 9db4 13b8 9885 130b 99db 98fc 96d1 1ba2 180e 9db1 1c2b 98d9 19c0 9408 9949 989a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 302 | 9194 9609 11a9 12dc 8f85 10e7 95c9 1991 90f2 9033 99b3 99fb 9902 1103 95df 9ab0 | 98cb 167c 9a19 99eb 1c0c 9d04 95ca 1238 9c0a 9c17 1247 1369 13ce 1ae9 97b8 12ba | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 303 | 92e5 0f0a 9844 96f5 0de3 8e93 12b5 12fc 9459 9a31 111c 9214 135e 1115 91ea 1173 | 1596 9a4c 1360 910f 9a77 1a36 95c5 95bf 94da 8f17 18b8 97ca 1749 970a 97f1 17c9 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 305 | 921e 1d1c 9a39 97b7 9a66 1532 1b5b 9bee 93a7 19c7 1273 12c3 97ef 1410 9108 9451 | 9ea4 13b4 18a5 1a77 1772 9a82 943c 9556 9d31 17d9 1ed7 9d34 19a9 9b21 1de0 1ab1 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 307 | 99dd 99c7 1320 90c1 1934 9095 18fb 93ff 8e97 1073 9481 17cb 11b4 931c 0eec 953f | 0fbe 112a 9517 17c5 114d 98fa 12f4 1654 199c 9b7c 9541 90d7 9606 18d3 99d5 9586 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 308 | 1908 95d0 9d24 9382 18c3 1289 180f 9604 15f9 19d7 9381 972c 9811 9cf1 9908 91e6 | 18da 1aac 14a0 9b4f 9973 9c70 9656 9a0f 98a0 95fb 9b67 9a0e 97f5 1559 955a 1e07 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 309 | 120d 1114 10fd 18c9 9c4f 9454 131a 17c6 9b33 19dc 1a43 16b1 95e8 1afb 18a7 194b | 1cbe 1982 1c0b 1370 9291 968c 1983 96d6 9240 9416 9178 9871 95cf 9409 123e 1593 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 312 | 8f99 9a39 9739 9352 1a7f 121a 8f64 141f 9026 1714 1a0d 9847 9610 16dd 9149 90da | 1a99 104d 139c 18b5 1203 1a72 1be7 16f7 1994 955c 125d 9340 93be 9389 1ab3 1bc1 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 314 | 1d5f 1864 9daf 9c2a 9e25 956e 19b8 149d 9815 95b8 1a4e 1548 9a68 9599 9d4f 13d8 | 92a0 16a0 935a 1383 1320 1937 93b2 9ab6 1641 1af7 947e 9a06 175d 19f2 93d8 9c60 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 315 | 9665 93d2 92a1 9ca9 1268 9cad 13f8 94b9 96e3 1a8f 19cf 1b35 1319 9407 1ad8 19ee | 9b1a 9af5 9d9b 91e2 1e82 15c0 1cea 9a0f 18dc 96b6 173b 9501 1c42 9bc3 970c 94ca | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 317 | 1aff 9429 1278 960e 9739 9274 96dc 127f 9606 0ff9 13dd 111b 9b8e 9944 936e 9139 | 9362 96a5 1bdc 1516 14c7 1aa4 15f7 192e 13bf 9ad4 9972 1b09 1087 91be 18a7 1a10 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 321 | 9669 9366 117c 146a 12ea 9137 92d1 1932 15f9 15c1 1878 93f4 199f 1a43 9a89 90f0 | 1561 98c8 9bdd 16b1 97d5 1b0a 989f 91c6 945c 9533 1184 9773 0fdb 1126 900b 1c09 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 323 | 91d0 9361 1740 16bb 9b5f 91cc 15b8 9271 14a1 1164 15ea 9094 96fe 182d 99f2 937a | 9c0b 19fc 1631 14d0 12f5 1c19 190f 9a30 9a64 9d00 976d 1b8f 97a8 932f 12cb 9913 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 325 | 95ed 1366 1118 1541 1af0 1191 9045 1718 14b0 971a 9914 0ee0 91fd 158d 9268 91e1 | 1616 96e1 1ab2 15fa 0f99 9928 1a9e 14e8 95cb 93ba 0fa4 9a5e 18be 9430 19e5 96e2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 326 | 1645 1264 97c3 1803 144e 1c8f 1b56 1c3d 9c05 1934 99f7 1401 1cb3 9679 937f 9a4e | 9713 9e71 187a 978e 9b0b 944a 125d 134e 12a7 1568 145f 9b64 93c6 98ea 1a11 967d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 327 | 150e 8f30 978f 9465 1988 9ac6 19a8 1105 9a7a 11d4 1a73 1a6a 1243 9800 12e5 93eb | 94d1 9ac5 13ca 990b 0fcf 9011 9250 1a6d 11ca 19a1 92e0 92ca 9b27 11a1 9927 19e8 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 329 | 9bf5 93cb 1ed6 9d50 9239 1332 1b43 16da 18c1 1620 1781 9ba1 9b77 9314 1c2c 1682 | 95d5 9e1e 940d 95f4 9ef5 1e6d 14a7 1b6f 9a3e 9cfb 1a9a 9599 96b9 9d9e 138a 9c0f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 330 | 91d6 1994 91c9 143c 1c87 151d 1ba9 96ba 194d 94c9 9c01 9995 16de 14ee 142a 9bf3 | 1b5c 16ea 1bc6 1bfa 93e9 97c7 9180 19b4 154e 18d6 91e1 130c 1a22 9c28 9ba9 14c5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 332 | 1775 9237 19cf 194b 99ee 90d3 17ae 962e 1830 13b6 14b7 9582 92c6 93a3 1a3d 10c1 | 15fd 1888 9317 9412 1287 9960 93ff 94c1 126b 9804 9530 1717 9ac4 97bb 91e7 9b52 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 333 | 15be 1522 94af 9145 9851 907e 163f 99a8 94b1 9982 195c 97e6 0f2f 95f9 1428 96cd | 1188 13d1 1358 9736 0fe1 1a43 143b 9091 16c0 9179 904a 111c 9a84 9327 143c 14ad | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 334 | 14ff 1503 14bf 16bd 12ec 9578 107a 1296 9793 1838 93c0 96e1 18b2 1099 9ac0 93b9 | 9ad1 19b8 99a3 986a 9c13 18bd 1bb7 1c02 138b 9556 98f0 1583 1374 1b25 9202 9831 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 335 | 1637 13ff 9563 1839 8fd0 9a18 0fc4 1491 969b 95f9 92c6 14ea 9636 90e5 8f30 989a | 1681 18f5 95fe 9640 1b34 11a6 9b88 96a8 1787 1563 99e9 9927 944d 9d28 9b10 126f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 336 | 1915 9bbd 9a7c 9aae 95ef 19a5 9c4b 173e 9a50 1d5c 1e2b 96c9 97d8 1454 1a27 99ac | 193a 158a 97fd 95f9 1a8f 16cb 1560 98a4 9857 92d1 9525 1a3e 979f 1d02 97c9 95d3 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 337 | 9188 1899 97c5 1806 98cb 12c6 1891 9c28 949c 1048 162e 12ee 1a71 92d5 8f25 17d5 | 1a42 14b0 136a 1452 12f9 1948 13b7 128b 96ef 1b5a 96b9 99cf 90ea 191d 9bdf 16b9 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 339 | 16aa 90a2 115a 918f 1984 971e 14dd 1a69 95e9 1231 9669 1ce1 98c4 181a 9689 99fb | 9701 1c2e 1c87 191b 122c 17ea 98ee 93a9 9770 9902 951f 11f4 1263 133f 94b4 11f6 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 340 | 935d 9645 93da 1134 14fe 9842 15d8 96ad 92aa 9145 8f72 99d0 9620 984a 9900 1504 | 1afb 942b 1778 1b9a 952e 9241 9707 9340 1b78 1997 1ba1 1339 94dc 14e6 1359 971b | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 343 | 9366 151d 1568 9365 1772 1a9b 9544 1535 16b7 194a 8f75 11fd 9aca 14eb 9852 9641 | 19ce 9424 978b 958b 14a8 1057 969e 9385 9593 1259 9b98 19fd 1278 145e 13d0 1354 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 344 | 93ad 12c1 97be 91bc 18a0 9a20 97ed 15f3 1b55 1529 9bb1 17a0 1440 1786 1414 19e4 | 9907 9a56 158d 9d18 9559 95df 184b 1924 14e8 98be 1113 15ba 9962 96c7 1a75 96fd | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 345 | 93eb 9af5 9b17 90e5 1364 9436 93e1 199f 9a3f 9a0b 1729 1a5c 1635 15bf 9649 98d4 | 15a9 1118 90e2 19d2 169c 17c4 95b2 929e 11ea 0f26 9172 0f67 954c 9590 9418 101c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 347 | 970d 1000 97e3 90e1 9abc 9492 9a25 955e 1524 9167 9218 1001 147a 1283 93ff 0f5b | 947b 1bf5 90a5 1baf 0f9f 950e 0f8e 95f3 1575 1aa5 16eb 1835 9860 19d1 15ae 1b61 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 352 | 95ec 1592 13b7 18e7 951f 1c1f 9c24 14f1 135e 15d3 9cde 13ea 9659 1572 1e18 13c8 | 9c94 1c4f 9ee7 96f8 1a37 93a6 1259 9d71 1db7 9c50 146d 1dac 9a9a 9a57 93f0 9e6a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 353 | 8f3a 9abe 900b 96d4 18ec 97e5 17ad 1527 15c4 9115 179f 98c8 1a70 9a82 9291 9054 | 1be9 8fcb 9cee 963f 129a 948c 12c8 98ef 96cb 1c93 94ce 935b 9240 1209 99e3 9a30 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 355 | 9271 98be 1abb 1ad1 9956 97dd 1236 190f 937e 15d4 9b73 1d61 1a3e 1778 1500 9b91 | 1bf0 190b 1659 9348 95d3 18c1 1d88 16fb 1e09 9a44 9548 93c4 96dc 9ad5 989f 1534 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 356 | 10b4 18c2 1a0a 1039 90cb 1a2c 1656 0f8c 9894 185b 15df 988e 14f5 108f 9b8f 95ff | 9ab1 93c8 0fcc 9a81 991a 8fb1 9446 1bed 94c4 923f 96ea 14a6 96f0 98b1 913d 9648 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 360 | 9af7 9bd4 1a47 159f 92d7 1b3e 1c6b 9266 9bf8 98e0 1a7c 1595 906f 164b 9845 93f1 | 158a 9350 956d 9806 9b1a 914a 12de 9ad8 9204 17b0 13c5 1ad8 9c72 97e1 9778 193d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 362 | 14f4 9aad 929a 9128 9a80 9754 148c 1243 1639 179f 98dd 196b 1b49 183d 9681 9903 | 9b48 93bc 1c4a 1c22 95d9 1916 9cdf 1c68 1905 993e 1778 177d 120a 98c1 9973 156c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 363 | 1999 1362 92a5 17d2 948d 1bac 143d 9d4e 1b61 97ce 9c19 15ce 1bd1 9705 18ab 17d5 | 9688 1e92 1d66 18a9 1bbe 96d6 1bf6 953d 1733 99ca 1375 9c40 97dd 1a9a 19e2 1b35 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 364 | 1968 1403 9600 9d98 1959 14ad 16b2 94d1 14b0 18f6 9a52 91ce 9b10 1c38 9b2b 1c82 | 93d8 9913 1772 12e3 9523 19eb 97cb 1824 9b10 9576 93ed 9c08 91e7 1294 1451 938d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 365 | 9485 9b0d 1522 1b20 9554 176a 9b17 936c 1377 1a09 15a3 133f 93bc 1c3c 9c60 9a1b | 97f1 1538 999a 941e 1701 17fe 927a 19eb 99da 1210 98ef 1ca1 9caa 937e 1462 149a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 374 | 90ec 9896 198d 15a3 9a7d 1409 9b43 137c 9b86 13a2 98fe 16da 12d5 91c8 92e2 9111 | 9c82 157c 93eb 15bd 909c 9a4e 11b1 9a95 128f 97d3 1490 93c8 98f5 9cef 99f5 998c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 375 | 1191 9457 1ad5 1112 0f6f 9b20 182c 9b0e 8e83 1185 8f2b 946d 9019 9a55 9172 8fde | 1aab 1599 8ffd 19f6 1a64 1233 1347 113f 9b59 1842 9a32 96ef 1a02 1010 9a87 1c54 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 380 | 1b6e 13f7 950f 1b93 9b50 1623 143d 1652 1358 198d 12e2 9974 9ba8 1c05 9c3f 971e | 1222 9ac9 1ab6 1273 92b0 9906 18f3 178f 9de0 1364 9e23 1635 92f8 115a 92dd 969c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 381 | 1353 19ba 9ce8 94bc 97ec 196d 9924 9359 1bc2 91f2 1882 96da 904e 13d4 130a 94f7 | 9d79 1445 130a 9bff 17a6 9721 15ba 1a0d 13c2 9d6e 9600 992f 9ca8 1a5d 9bb4 195f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 385 | 11c5 13a4 98a7 1bbf 9b43 91a5 9558 1878 18a5 921d 171a 17f7 168b 9a8a 152b 9c7d | 1def 9db3 15d0 1283 1731 9d6e 9a62 98ed 9939 1c9a 9972 973a 1a96 145b 1b45 13a5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 386 | 1c83 164a 982d 1479 95cc 96ed 1437 1927 1d0e 9784 9c1a 1332 9c65 9323 9287 147e | 9409 9914 16f7 1905 97af 1914 19b5 1791 947e 9990 134a 9c83 946b 1a60 1eb0 1b7d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 387 | 1031 0e48 150b 91f3 96de 9b15 9a94 1354 10cf 1184 98ce 9558 9830 8fa3 16a2 1763 | 194d 19bf 148d 1a39 9291 105b 0f8d 984c 9b0b 1683 1276 1660 11a0 1b1d 9520 90b8 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 389 | 1248 97e9 969f 1b18 9aaa 1977 9b0e 9389 9939 98cc 15bd 9794 1c58 183f 996b 9b14 | 9e49 167b 1711 1314 939e 950e 1505 1d25 16e1 95ef 187c 1646 934c 95ac 97d5 15b5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 390 | 1b03 190e 92a7 1c93 1b7c 9682 9b30 9c41 949a 12df 99ef 15ff 9c6e 91fc 1ad2 1a03 | 95eb 15c9 9e04 94bf 95a0 1a7e 96d2 14b1 1a9f 1e59 9478 1876 12b3 1e71 960f 95a1 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 391 | 13a8 15ba 9323 153c 1b27 93f6 9b7b 905f 1b4d 1a11 14d6 1860 9bdf 98be 9719 15fc | 183f 99d8 9863 986b 1284 9ae4 90a7 9b2c 1364 119f 99dc 1541 11d7 13fa 143a 16cc | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 392 | 9295 9798 9c74 1652 9544 967f 9a7a 1d3e 164a 99f6 1477 1754 9a4f 986f 9be3 1341 | 1e34 9669 12f6 9929 98ef 1787 1407 125b 9b6a 962e 1aa9 1741 960a 1582 957d 1bc6 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 393 | 9d2b 9873 9890 1677 1dcd 1516 1786 1457 9a6a 13a0 1919 9498 9c93 1321 9b85 98c9 | 922e 9aaf 9711 99f3 1287 1d1b 17fa 9b33 182f 9e4b 1796 1b36 93c5 9f43 161b 197b | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 394 | 9b46 1d66 9dbc 9929 197f 9252 1b9e 92f9 99a9 99fc 9a7b 133a 9713 9aca 13cc 91c0 | 96c4 93d8 9356 97a7 977b 9cfc 95ad 1f2f 97c5 1725 167f 1b30 96ef 1624 9d9d 1ca6 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 397 | 97de 15d6 9ada 13d4 9432 11c5 9502 1a4c 10bd 1aed 99c0 114f 1ab1 978b 9969 1267 | 16b3 9776 140e 9c22 9a49 9b44 176f 1573 1b6f 14c3 94f4 1b7b 1263 14e0 16ee 1a9c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 400 | 137e 9b51 9cde 182a 110c 18eb 932a 15a4 1536 1c1b 1a15 9cad 18a0 92a1 1a97 1382 | 9cf3 146f 93f5 953a 9ca2 1847 1e28 9af0 1b11 1380 96eb 1492 9504 1e35 93c4 9ad5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 402 | 1b28 95d6 9be5 1678 11d4 1d5c 1be9 1b5e 9a63 9b5a 1cb9 98af 1343 91ee 9b3c 90ef | 1366 9900 93c2 9a2d 1c24 937f 10ed 94f3 1311 1267 91d6 17be 9cf9 1b08 9314 9c07 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 404 | 924d 15ef 93ad 972d 8e30 96ce 0ee2 97e1 19e6 8ffa 9173 1026 1279 0e1b 91ba 191a | 16e7 91aa 94ae 9061 9a79 1447 1acc 9073 9029 1842 9817 198f 971a 1afd 9732 91f5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 405 | 1545 9a35 18c0 948e 9573 1954 9639 1ccc 141d 9881 984f 941d 14a0 1a5a 95dd 9701 | 9cbe 14b2 179e 1bc0 1c1f 96f7 197b 91f8 9a75 96e8 99e6 9a38 1b34 9623 1c18 9a5d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 406 | 927b 1a1d 9a5c 9263 172d 10c7 9584 1b3d 125a 1074 9b84 97db 9122 9344 9757 936b | 9a38 93f7 13af 18c0 952d 1a9a 964e 125d 9a60 9a9b 90e0 94f4 9c03 1a88 947f 991c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 407 | 1452 98a8 1339 158c 1299 1b53 1a56 8fe3 18e4 14e1 9b34 174a 9576 117f 98e2 93cb | 1826 1563 1948 9650 1808 10a5 9095 1b93 13d0 9924 12ee 9544 14ca 9a8a 1134 9696 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 408 | 1c79 9810 1972 1543 16e4 9bae 99a1 11f0 17d2 1795 1225 95a9 1877 10d6 9cfd 1906 | 91a9 1697 9375 992e 1525 9269 14fc 1a49 93fc 9691 9aaa 1910 96c1 9a66 91e9 9574 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 409 | 1416 189e 9555 18c1 1673 1774 1a2b 1908 959f 123f 98a6 9c10 1431 1b37 1566 9b09 | 9ee7 96f6 9c2b 992b 99e6 9a77 15e4 9852 994e 9ee6 9753 14b2 9d3b 1738 9aa5 95f4 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 410 | 93ae 9dfb 154b 1b7f 9a8d 96ef 1828 1471 19b2 9cd0 95ca 9a50 1bf8 937e 9dc8 1c9d | 1c83 944c 1d79 937d 97c5 1a67 17aa 9c60 14b8 1330 1a85 1531 94a5 1ec0 14a0 95d4 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 415 | 927e 1b26 19ad 90f9 1374 9a76 9b41 9b14 955f 1381 94e3 1420 953a 1b1b 93f7 1408 | 9e42 1460 15e0 9c93 9e54 175c 15bd 960a 9c4c 1ac8 992c 9982 189b 95d5 9c88 19e7 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 419 | 93f0 19ab 1ddc 9622 9645 1a50 1be4 1c41 1500 9113 9509 13d5 14bb 15ef 959f 18fd | 195b 9634 1298 17af 1621 13a8 921c 91e3 9b07 1d3b 99c4 98dc 981d 19d9 16c2 97d9 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 420 | 1297 983a 9912 9791 12ff 916d 9278 939a 8ffc 94cd 9848 10bc 9066 1774 18a1 1372 | 9944 0fff 12e5 10ef 15ce 9970 96c5 9401 19d5 158e 1049 1a67 9b19 926a 91b3 986a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 421 | 9433 1764 162a 9a79 98b6 1b81 9a02 1c5b 19a5 98cb 1121 92a7 9884 9322 918d 9650 | 1a7b 93e3 95b3 114d 14c1 12b5 1120 9289 9123 1442 199b 1c6b 94ac 98ae 193d 9626 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 422 | 0e48 913f 1ac4 929d 18c3 10a0 154d 137c 91ce 9233 1377 17c5 8ffe 8f51 91d1 169d | 9ac0 1999 1006 1681 9181 18b0 9610 17ca 1a3e 98d7 9592 924d 9a35 1b82 9a81 95c2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 423 | 9bd1 9a82 9c67 9565 15f2 9651 1345 1514 16b6 12e7 9aa3 9cee 1a56 166c 142c 9a5a | 1281 92e7 13f0 170b 96ad 99ea 1957 96cc 98f2 9a0d 1352 1314 9304 96ab 17d0 934f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 424 | 11b0 1897 178c 101c 14a4 8fb7 14b8 98a3 1064 1be5 11ee 15d1 9802 9690 91c8 1935 | 1cc2 1336 9440 1c5a 9a4e 9c60 9951 1329 9cba 936d 9ccb 1793 1406 9508 1997 13d5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 425 | 13d3 9200 1a25 9244 9b6e 17da 1455 9211 10f1 9957 97e3 1b3e 18da 9b95 9629 9a2d | 9a44 1cac 92fd 9c38 13ad 9483 9a3b 9b36 1d1d 1571 15b2 9446 151e 11a5 9595 1183 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 426 | 99ab 998b 10a4 1b30 9850 9a10 1217 96e0 1b83 1758 1702 94df 99c6 9216 0fac 91b6 | 15cc 926b 1d01 9019 9467 114a 1c88 94a2 92f9 1680 16a3 1ac2 9207 1a4e 1bbf 9a29 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 427 | 9205 1417 9aac 141f 9634 1c80 1ade 944c 1207 9697 9b43 183f 9288 136e 9a70 9cb1 | 1ad1 18da 92c4 1a08 9894 9479 956e 9b14 1e98 9853 151f 94e7 1a68 1bb7 12db 124e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 428 | 93cf 9441 979e 9831 1495 9a88 1321 1342 1056 90b6 98c5 0fdf 964a 955e 1b71 14ec | 95ce 9744 9111 9124 9510 9212 98a9 1860 9904 9c1d 116c 1ba9 16ed 940e 91f6 9559 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 431 | 931a 96af 9c11 14a7 98de 9b00 1c76 9c86 1e0a 1dd3 9e42 9cfc 1405 9a96 9282 13cd | 1bf5 1913 1307 19e2 98cd 1665 1553 1474 9306 1331 13d5 91c8 99b0 93ce 1ca7 9e67 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 435 | 9ad7 9705 930e 915e 9697 9579 9411 997d 994a 9497 11de 153b 1c5e 9ebc 920b 19ae | 17ae 1a04 1ede 1d11 180c 19d7 9d48 1934 9662 1a68 9d53 9960 1527 93c4 1db1 1874 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 436 | 913d 1567 9a3b 99d8 9889 9060 1ad6 1a0b 1c05 1c64 1248 9618 9184 1bb2 90ad 1c75 | 1cc5 99fc 1179 140c 942a 9ae1 941c 1451 91ab 9157 991e 185f 9cbd 1275 9d13 927f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 439 | 9701 96f1 9353 9c6f 1b0f 9d6d 974e 17fa 1af8 19a4 964b 13d2 999f 1752 1b6b 1980 | 9b33 9ab4 9b8f 12e2 13c5 935a 1b6e 19d7 1504 95fd 9a4a 9f6e 1663 9b90 14c3 973d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 441 | 95f8 0f4b 94c6 9070 0f6c 1797 10d5 964a 1388 1197 967b 1270 936f 92c7 157c 95f8 | 1702 9bdb 94ff 994f 99c5 14f1 9a6c 95f5 1767 194c 9546 9a7e 9754 1a15 952d 960e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 444 | 1ad1 9715 0fdf 9101 8f0b 904b 9709 12a7 9a61 927d 917c 987e 18e5 903b 0f3e 1086 | 8fe6 93d8 190b 1a6b 9ba0 9bc1 1449 95e6 124f 96f7 9a43 9034 93b6 1824 1ac3 183e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 445 | 132d 182a 19c9 9457 95e3 9572 16a6 96ac 1092 970e 939e 17f1 91dd 15f8 1364 142b | 9882 127b 9262 14ab 1594 13d6 153d 1325 989c 93ed 95d2 13cc 1a14 9646 98cc 1784 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 446 | 9bbd 1970 1272 16ea 9b68 9292 951d 15be 1d62 9675 9b7c 93f7 92f8 193a 178d 1be6 | 14bd 9594 9e7a 98e8 171c 9db6 9d24 9a3d 12f7 9c74 93c6 1f72 1ecc 181c 1b74 948d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 447 | 8dc2 9806 995f 14d4 9516 111a 12c3 14af 185f 1145 131c 8e73 932d 16a4 9aa3 0d83 | 9a47 917a 8f94 96b0 9578 986f 15bb 9665 918e 9723 15ee 9ae4 976d 134a 90cc 1a1f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 450 | 9969 9145 16ac 96b6 98a9 1733 1604 934d 11d4 19a3 11a2 9897 1459 9844 180d 99db | 958e 1b87 95e9 9890 1567 994b 19ed 19b0 9e40 92b0 1c5c 15d3 9b8a 14c1 1507 15b6 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 451 | 1866 93c5 908c 1720 939c 98ec 9308 9b6d 9396 184d 12a0 9ad8 970e 9728 1a39 9ae4 | 1784 1cec 9bc4 95ad 9c22 9686 1d2c 92af 9929 94b3 1d6b 144a 1570 9842 1373 945d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 452 | 97cf 9403 1352 9677 999e 104b 1098 923b 15a8 963e 94c8 17b6 1712 91d1 121f 1064 | 913e 9752 96b2 131c 8fe8 99b1 9a47 179d 927b 94ce 930d 1267 11c5 1a05 9644 998e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 453 | 19ef 1921 948d 1191 9831 9412 137a 9454 9258 1973 1759 15c6 1648 18d7 176e 9735 | 1315 14cc 9ac7 9c90 15d5 9bae 9bb0 1b51 1beb 96c8 188d 97c3 9a69 1683 958d 180a | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 456 | 1424 987e 9361 9938 982c 13b3 15ac 94b5 17a6 9217 9aa1 1104 10e7 1013 9b67 9773 | 17b0 12ff 1bcb 91a0 92dd 18d8 967a 19b8 1518 18d1 9230 1aab 1caf 1bfc 904b 9480 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 459 | 15ea 173a 1812 98d3 1594 96a5 124b 994f 1525 8fea 13f4 9ae5 9ac8 9059 1ad9 13c3 | 181b 9662 94e0 149d 97db 141f 9af6 925a 1710 1b59 17c5 10e3 1061 9bdb 9241 197f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 460 | 1b07 9d61 9bfd 18f4 9883 9a07 9c77 1942 1b73 9721 19d8 1b0e 1b71 9be4 18ab 1303 | 1504 1486 93d5 1835 1733 1610 1558 16ad 96f9 1b6c 18ef 9351 95a2 13fe 196e 1c13 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 461 | 1670 9176 16d6 195c 93c2 9066 9b27 1125 1455 102e 1692 12fe 16ce 14e8 18af 1a92 | 96dc 99fd 9376 941d 185c 1ada 90c7 9bbe 9513 9c6c 95af 1744 92a0 9738 9171 915d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 462 | 9c3c 939e 1b76 11e1 136f 13c1 9bd7 198b 14cd 9b33 92c8 954d 9729 1290 98e9 1338 | 1194 98ab 9118 9b84 98b5 1a5c 1337 1305 972a 1391 1bd6 98ef 9446 1b38 13de 9810 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 463 | 175a 93a0 18fd 19d7 923b 99c5 9d84 19fc 1ab1 1469 9a50 1996 9dfc 1d40 9b62 12a7 | 96b6 9b63 1690 17b1 1d1c 97b0 1297 15ef 1523 1d95 17d3 9618 1472 147a 14f0 1b20 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 464 | 12a9 1066 96d9 9c36 1731 978a 1afd 1a15 91f1 95ae 9b78 9692 939b 9ba2 1a31 9432 | 1abd 1c70 96d0 91f7 1572 15cb 12f3 132c 1a19 19af 92b5 9516 9aea 91e2 9320 19fb | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 465 | 963c 1117 9864 1c0c 9107 9945 98ae 1514 12c2 9779 97ac 1022 1920 9406 15ca 167d | 1a68 1bd7 158e 127a 1b8e 944a 9681 978b 1bca 9781 161a 1c33 150e 9a89 1975 997c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 466 | 9858 94ca 14a9 9a39 1991 1676 1706 9c23 18ee 116b 1565 17d5 9a9f 9b2b 92bd 1ba2 | 98e4 1c9b 18d6 1569 175e 9a31 97d4 911a 9616 9c72 9b63 987e 93d0 1587 9af0 95f9 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 467 | 1c6e 9adc 1b7c 9614 1890 154f 909d 9437 9481 13f9 9476 987d 1af1 94f2 18db 9da1 | 919a 1290 11d9 1750 964a 1aaa 1c02 995c 9a0d 1885 1b7f 1775 1546 99ad 143e 12b6 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 468 | 16ae 9590 1871 92e7 979d 1926 0f57 169f 1264 14e7 19ae 16b6 16b1 92cf 17d8 99ac | 953f 1371 12fd 1719 9295 11fb 1aed 14b0 9ae6 9435 8fcf 1419 13a5 1765 1113 0f6c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 469 | 9286 19ad 977b 965e 19c6 8f80 90b8 1619 93d9 128b 9a6f 12fb 93be 9736 9311 1899 | 9945 1391 15ef 9582 939c 9c61 1acc 1854 1879 9c96 1270 9acb 1ad2 145d 1b19 13ca | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 470 | 99d9 978f 13df 95de 9a06 122e 9309 12af 19ac 9514 9b90 9d18 188b 9105 1a51 15d6 | 92ae 944f 9c0f 9899 92d3 9a33 98e2 99e2 165e 173e 12a0 12ec 16f5 1adf 14d9 99c0 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 472 | 1107 1bc8 1310 956a 9a6c 1709 9923 1720 9268 1b84 9638 995f 9cd1 1697 9ad3 92cd | 9ad3 9132 1a22 161e 1230 9753 94f9 140d 1972 12c3 1506 95e7 125d 9871 91e5 1944 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 473 | 9477 9d25 93b0 13e3 931a 1a9e 9ad9 1b86 1929 99db 97fd 9247 98a8 98f0 12c8 187c | 1bc3 9371 1ea1 9bd2 9e44 963f 96fd 9494 18a5 9864 1b43 9ee9 185f 97b0 9eca 9968 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 474 | 938e 0fc6 1474 9a9e 95ee 93dc 13bf 9873 959e 12aa 1266 173c 97c9 17c7 99e2 0fc3 | 1652 994b 9365 1067 95eb 95d8 93f7 12e8 15f0 17a8 955d 10e9 936f 1239 10d7 9954 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 475 | 961b 1718 8f63 8e90 0fe8 18de 1038 11e3 9044 8fcc 14d3 926e 12c8 11ec 0fa4 98ed | 15f4 91cf 1b49 991d 1c3f 1306 1971 974f 9989 1c10 15be 16ba 9963 98d5 9c04 91b5 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 476 | 9211 1b95 14ea 1690 1a83 1bfd 920b 1292 1d24 9225 9376 9210 13d1 1c59 9383 115b | 9af3 91b6 9884 9894 92d6 1160 9e72 9d8d 123d 9db8 191e 1e4d 19de 9451 1c09 1c4f | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 478 | 918e 99e7 0fae 9a0b 172d 1169 9936 1111 957e 0f97 0ee5 17a3 104f 9270 9785 8ff3 | 1800 0f23 1840 9071 906c 9a66 0fd9 1631 9539 17de 193d 106d 18e4 9509 9321 17ea | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 479 | 9ae2 967d 13ec 1453 9428 950c 9216 1aed 1b71 1320 18fa 9690 132c 99c3 9457 930d | 0ff4 96f5 994d 96c1 1543 1749 1a42 8fe9 10ca 16ed 93a7 964c 995f 919f 9500 970e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 483 | 91a7 1944 19d8 1a36 11c5 9863 903a 11bf 19f2 1157 1195 9432 945c 179b 9106 1977 | 1c87 12fe 915c 9279 1cef 1409 1a3f 9a60 1226 9a9b 1956 1890 9831 966f 1bf4 93b6 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 484 | 9568 9515 9360 97b6 9177 9724 9061 1668 1323 1785 903a 9799 91e4 991e 9655 95a2 | 96a8 159d 1766 9354 9882 13b7 1bd7 9698 9909 931f 18c4 91eb 97f8 1168 1613 15fc | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 485 | 1578 1669 187c 92b5 9553 1863 1185 16b5 18d2 1567 95f0 9615 13e4 0ee4 1731 1768 | 95fb 1515 8f3e 965f 9367 0fcd 9966 127b 1045 92eb 9391 9441 139c 9b61 940c 90a2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 488 | 9093 913d 11ba 1824 93ff 19aa 97cc 185c 98b9 9200 1516 9ba8 19b4 9146 134d 9ca6 | 9d32 1c80 1cc6 943e 9958 96d2 9656 9794 15b7 9af6 983a 1257 92ad 1c20 9bff 91c9 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 492 | 1a2c 17dd 9ad8 950b 9d84 1be4 9541 1c88 1c5a 1622 963b 98c1 915f 13c8 972a 9354 | 92b7 9714 93fa 1ac4 92ab 13ca 976d 9374 9474 183f 9702 14f2 1cda 9a98 9675 19d3 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 493 | 9ab5 18d6 9258 1209 1ac5 160e 9385 9b39 928f 1633 928e 9b73 9784 1812 9521 1869 | 9296 9547 1e36 9d1d 12f1 19e9 99c2 1269 9b79 1985 9a89 9431 9594 979f 1a40 95d9 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 495 | 96b4 9a8d 12a9 99a6 1ba9 17f1 187b 9c81 15ff 1d96 1526 16f0 1195 9d34 18a0 1cc6 | 99d7 185f 1f41 17ae 9705 195c 17d7 145d 9974 1487 1afa 1966 9d0c 958e 9651 14bf | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 496 | 9873 1674 1b53 9788 1346 93dc 9b83 13e2 15a9 97d8 1d16 1876 1dfc 1330 956f 926e | 9794 1a92 1304 96ab 1f59 9cb7 1429 9e20 19f6 1a91 92ab 9a36 928a 9e61 1bfd 1dd1 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 497 | 14e5 14df 9208 90ae 1984 9a3b 187a 1735 9039 91aa 9a52 9b8c 0fc7 91e2 16fb 9484 | 98d4 97ec 990a 1adb 1004 1187 91c8 14be 9999 9c2d 8f91 1102 9bb5 9856 157e 1745 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 498 | 171d 91c5 1c5d 1136 1967 933a 9a7e 94d2 9c49 9863 9521 14fd 97a6 15c4 11eb 1a00 | 9777 9a7b 11c9 9a70 965c 9929 9485 1807 12de 9302 16f1 9a76 978b 16b0 1cc0 1327 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 500 | 995a 15b2 9337 951c 15dc 1620 98b0 10e4 18e6 92b8 19c5 957b 0f9a 8f84 1a5a 10c2 | 1568 9505 998e 166f 987a 9815 148c 9c42 9375 99ff 12e4 178a 1c4c 9b74 92a5 9d29 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 501 | 956c 9876 9916 1575 1570 17d5 1be0 984a 12a4 1027 921d 92e5 171d 12e1 1951 99a7 | 1904 14d3 14cf 9714 9903 150a 924d 92ab 9b59 1a6b 1989 191a 962e 996b 13f8 117c | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 502 | 1115 195e 1440 13d8 9b13 98be 16b7 1ad9 94c8 919d 17b6 123b 9894 1396 17d4 9685 | 9b86 95da 1921 9b3f 1335 131d 16cb 10ba 9929 1b60 13ba 1c2e 16bd 9bbe 9737 9777 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 503 | 148b 1457 9348 95b3 96d2 8ee2 178b 931d 1530 9a09 99da 90b3 9214 188c 9046 1a6e | 14a9 148e 19f8 9484 9399 9aa2 129f 18a2 9491 0f55 906a 9b86 182c 139b 9882 91e0 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 504 | 9711 96a4 14a9 170a 9a8d 97f9 1b5b 171d 979e 92c6 9892 13b1 9889 11c2 9136 15c1 | 1453 135c 177b 94c0 114f 91ed 905d 13a3 92ea 9790 9252 1648 12fe 1904 991a 95cf | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 505 | 1a93 94c9 9973 97ba 177c 159e 19f7 104b 143d 17e2 9243 1a4e 17de 14b1 98a3 9421 | 11a0 97de 928e 1375 917c 15b7 92e7 99e3 97e3 910c 19e9 111d 11f9 1526 11f8 9753 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 506 | 175c 14b2 0f90 91af 960c 1153 15c2 9a05 1b99 9188 94f7 1189 94eb 16f5 13bd 8f53 | 125d 95a6 1b8c 1bcf 14c7 197f 95b5 92cc 913b 98ec 159e 9a64 94ad 92b0 17c7 1a57 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 507 | 9388 997e 96f4 1462 92af 989a 1a8a 1380 96f2 1042 14e0 1385 0f50 91b3 1094 176a | 1547 1247 9372 1434 9506 1175 1051 982d 93ee 9b99 93ef 17eb 99d0 97e8 996a 1166 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 508 | 9b8c 9986 1932 0fa4 926b 1944 1bcf 1473 98e8 91dc 1176 90df 9549 1b30 16fc 137c | 90dc 9136 919e 9a9b 1a4c 9471 9119 9784 147a 9907 98b3 991b 16be 126f 9609 1a79 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 509 | 927e 954a 9623 1496 9a75 13ce 902e 955b 9a2d 0f9a 18d8 15b3 904c 173c 910e 0e81 | 190b 13ac 9371 937d 8f7a 967d 989b 1454 8fc3 9b89 0f84 9627 1878 112f 1908 1aa1 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 510 | 1cb3 96ad 9d30 9903 99cf 9425 99fb 99bb 10f9 9a46 9b0a 921d 19a7 938f 1c12 9c29 | 9315 951a 922c 14d7 159f 1970 156c 14eb 9b05 1225 917a 9a95 9370 98f8 1323 11c2 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | underflow | 511 | 9d44 9aee 136d 179c 99a3 16de 9e8c 1dad 1370 945e 949f 150f 1463 1630 99d5 9ba6 | 94a1 94a3 1e80 9bfc 97b1 9c37 147d 9286 1ff2 9bed 1d85 1d47 9f76 1b15 978c 1602 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | sub_boundary | 13 | 9557 94ff 1e5c 9d2a 979f 9792 1e7b 98a8 98c4 18a8 18cb 9333 18c4 9fe8 1e4b 1a07 | 9df6 2029 9666 975c 9c05 9a6e 1440 9a5d 99b7 18c7 1960 1ea6 1b22 95f0 975f 9730 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | sub_boundary | 95 | 1b3a 955b 1381 9e32 98e3 97d8 19c1 1892 9d73 9656 15ea 1715 13e1 92b7 9da2 9819 | 9639 1b6e 1d2e 9444 182c 1bd6 1794 98a6 1654 1ce1 1d4c 99c0 201b 9d99 94cf 9a63 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | sub_boundary | 187 | 1b0b 15b9 14f7 1663 9852 1731 1ee5 14c5 977d 146f 997e 9c80 1cb7 99e6 9bee 19d5 | 982b 9a99 9e5f 9b59 18e0 9945 13ff 1bd9 1cd8 1e40 99a4 1521 146b 97c1 16c9 9778 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | sub_boundary | 269 | 97b3 16ab 9557 962a 1bd4 180b 9caa 967e 192c 9746 1626 1661 1c22 9aa8 17fe 9735 | 9b4b 9b8c 9c0c 9c83 14db 18e2 1383 1d46 1885 1903 1c56 9abf 148d 17bc 9bd5 9a49 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | sub_boundary | 273 | 9742 9a41 9d61 97ee 97a7 96a5 1c41 9dee 147c 187b 12cf 99cd 1aa4 9705 1a0f 1f25 | 1b87 95bb 1592 18e1 9991 997b 963c 9367 9def 9a4d 9d9c 189e 1514 9cf9 18a5 947e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | sub_boundary | 431 | 17b7 9e64 197d 9ae9 19ce 9945 1dd5 1c97 9e71 9937 9f61 9856 177d 9e25 139c 15ad | 9aec 9339 9acf 1a4e 978a 9bd5 17a3 95dc 1464 1b1e 1489 1a32 9dab 9355 202d 9d7d | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | sub_boundary | 457 | 9a2c 17d5 93cb 18bd 15cd 1b10 13bf 1778 1afe 14ae 99d6 1d4a 1afb 1a1f 91c0 9c6a | 16f6 183d 1d1d 9975 9ddf 14b0 9d3b 19da 964a 1dc4 1a29 1621 15d3 962a 1e5b 169e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | sub_boundary | 503 | 98cb 99df 9ce6 9ab0 9c75 143b 16e6 1908 94d9 9638 1af0 9abf 1772 992f 1aa9 1c60 | 1741 9789 949d 16ca 95f9 9cb1 1a36 185f 1bb7 9a78 986c 164e 1a01 175d 97e0 9529 | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
bf16 | sub_boundary | 505 | 1622 13dc 9cab 9394 1bb6 1dfd 9ac4 19fc 96a9 1d03 1d43 9337 1ba5 1b55 9402 954e | 9c7c 1ffa 145b 9cdc 9768 92e2 96fd 187b 1c65 95ef 94a1 9eab 16fb 9843 9be3 9a8e | 00000000 | 00000000 | 00000000 | 80000000 | 00000000
```

</details>


## Reproduction

Run from this audit checkout on the RTX 5080. These commands extract the probe sources below to a fresh private directory, use the assigned interpreter, and remove the probes on exit. The optional harvested test plugin redirects the fixture in memory to the existing input. Run the latency probe with the GPU free of other benchmarks. `gridlimit.py` is expected to raise on the audited revision.

```bash
set -eu
umask 077
mkdir -p "$HOME/tmp"
test "$(stat -c %a "$HOME/tmp")" = 700
task_tmp=$(mktemp -d "$HOME/tmp/pi-c501-audit.XXXXXX")
trap 'if test -d "$task_tmp"; then rm -r -- "$task_tmp" </dev/null; fi' EXIT
export TMPDIR="$task_tmp" TMP="$task_tmp" TEMP="$task_tmp"
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="$task_tmp/triton" CUDA_CACHE_PATH="$task_tmp/cuda" TORCHINDUCTOR_CACHE_DIR="$task_tmp/inductor"
PY=/home/ryan/Documents/projects/ai/.venv/bin/python
"$PY" - <<'PY'
import os, pathlib, re
text=pathlib.Path('audits/AUDIT_cycle5_c5_01_gpt62_astra.md').read_text()
for name, source in re.findall(r'^### ([a-z]+\.py)\n\n```python\n(.*?)\n```$', text, re.M|re.S):
    (pathlib.Path(os.environ['TMPDIR'])/name).write_text(source+'\n')
PY
git diff-tree --no-commit-id -r --name-status 596b5222d0fd96fdb6d6ef6c6e6386464361ba0d
git rev-parse 596b522^ 596b522^^
for probe in receipt totals mma extreme historical devicebits gridcontrol hook checks capture syrk large latency repeat; do
    "$PY" "$task_tmp/$probe.py"
done
if "$PY" "$task_tmp/gridlimit.py"; then echo 'grid no longer fails'; fi
/home/ryan/Documents/projects/ai/.venv/bin/pytest -p no:cacheprovider tests/test_c5_01_syrk_and_mma.py tests/test_schatten_switch.py tests/test_polar_switch_blackwell.py tests/test_c4_03_wide_and_gram_drift.py -q
"$PY" "$task_tmp/harvest.py"
```

<details><summary>Runnable probe sources</summary>

### receipt.py

```python
import ast, collections, contextlib, copy, hashlib, io, json, math, os, pathlib, statistics, subprocess
import numpy as np
from scipy.special import stdtrit
from research.polar_opt import bench_c5_01_syrk_and_mma as p
from unittest.mock import patch
T = pathlib.Path(os.environ["TMPDIR"])
f = p.CANONICAL_OUTPUT
def strict(pairs):
    d = dict(pairs)
    assert len(d) == len(pairs)
    return d
def invalid(x): raise ValueError(x)
def read(s): return json.loads(s, object_pairs_hook=strict, parse_constant=invalid)
d = read(f.read_bytes())
assert d == read(json.dumps(d, allow_nan=False))
sha = '307fe7a8dc4fa7a9ee7642c26e33c42e6ec8293f'
def git(*args): return subprocess.check_output(['git', *args])
paths = set(git('ls-tree','-r','--name-only',sha).decode().splitlines())
todo = ['research.polar_opt.bench_c5_01_syrk_and_mma']
files = {}
while todo:
    m = todo.pop()
    path = m.replace('.', '/') + '.py'
    if m in files or path not in paths: continue
    files[m] = git('show',sha+':'+path)
    for n in ast.walk(ast.parse(files[m])):
        if isinstance(n, ast.Import):
            todo.extend(a.name for a in n.names if a.name.startswith('research.'))
        if isinstance(n, ast.ImportFrom) and (n.module or '').split('.')[0]=='research':
            todo.extend([n.module]+[n.module+'.'+a.name for a in n.names])
h = hashlib.sha256(b''.join(k.encode()+b'\0'+files[k] for k in sorted(files))).hexdigest()
assert sorted(files)==d['meta']['code_modules'] and h==d['meta']['code_sha256'] and d['meta']['git_sha']==sha and d['meta']['git_dirty'] is False
print('artifact',hashlib.sha256(f.read_bytes()).hexdigest(),'closure',len(files),h)
p.validate_results(d, canonical=True)
errs=[]
def check(a,b): errs.append(abs(a-b)); assert abs(a-b)<=1e-12,(a,b)
def paired(s,vals):
    v=np.log(vals); mu=v.mean(); half=stdtrit(len(v)-1,.975)*v.std(ddof=1)/np.sqrt(len(v))
    for k,x in zip(('n','geomean_ratio','ci95_low','ci95_high'),(len(v),*np.exp([mu,mu-half,mu+half]))): check(s[k],x)
def order(rs):
    first=collections.Counter(); second=collections.Counter(); prev=collections.Counter()
    for r in rs:
        a,b=r['order'];first[a]+=1;second[b]+=1;prev[a,b]+=1
    return tuple(c['native'] for c in (first,second))+(prev['native','control'],prev['control','native'])
def desc(v): return '/'.join('%.6g'%x for x in (min(v),np.median(v),np.ptp(np.quantile(v,[.25,.75])),max(v)))
bounds=collections.defaultdict(list)
ops=[]
for c in d['timing']['cells']:
    rs=c['rounds']; assert order(rs)==(20,20,20,20)
    for clock in ('gpu','wall'):
        a,b=[np.array([r[v][clock+'_ms'] for r in rs]) for v in ('native','control')]
        s=c['summary'][clock];ratio=np.median(a)/np.median(b)
        check(s['native_median_ms'],np.median(a));check(s['control_median_ms'],np.median(b));check(s['median_ratio'],ratio)
        paired(s['paired'],a/b);bounds[c['mode']+'_'+clock+'_ratio'].append(ratio)
        print('T', 'x'.join(map(str,c['shape'])),c['kernel'],c['mode'],clock,desc(a),desc(b),'%.8f'%ratio,*(round(s['paired'][k],8) for k in ('geomean_ratio','ci95_low','ci95_high')))
for k,vs in bounds.items():
    for fn in (min,max): check(d['derived']['timing'][k][fn.__name__],fn(vs))
for c in d['long_k']['opus_l6']:
    rs=c['samples']; assert order(rs)==(2,1,2,1) and max(c['shape'])==1<<25
    a,b=[np.array([r[v]['wall_ms'] for r in rs]) for v in ('native','control')]
    r=np.median(b)/np.median(a); check(c['speedup'],r); paired(c['speedup_paired'],b/a)
    ops.append((r,np.exp(np.log(b/a).mean())))
    print('L', 'x'.join(map(str,c['shape'])),c['kernel'],c['dtype'],desc(a),desc(b),round(r,8),*(round(c['speedup_paired'][k],8) for k in ('geomean_ratio','ci95_low','ci95_high')))
for key,vals in zip(('opus_l6_speedup','opus_l6_speedup_geomean'),zip(*ops)):
    for fn in (min,max): check(d['derived']['long_k'][key][fn.__name__],fn(vals))
print('checks',len(errs),'max_abs_delta',max(errs))
def probe(label,fn):
    x=copy.deepcopy(d);fn(x)
    try: p.validate_results(x,canonical=True)
    except ValueError as e: print('REJECT',label,str(e)[:95]);return
    print('ACCEPT',label)
def edit(path,func):
    def do(x):
        y=x
        for k in path[:-1]: y=y[k]
        y[path[-1]]=func(y[path[-1]])
    return do
base=['timing','cells',0]; raw=d['timing']['cells'][0]['rounds'];idx=sorted(range(len(raw)),key=lambda i: raw[i]['native']['gpu_ms'])[3]
for path,fn in [
    (base+['rounds',idx,'native','gpu_ms'],lambda z:z+0.001),
    (base+['rounds',idx,'native','gpu_ms'],lambda z:math.nextafter(z,math.inf)),
    (base+['summary','gpu','native_median_ms'],lambda z:z+.01),
    (base+['summary','wall','median_ratio'],lambda z:z+.01),
    (base+['summary','wall','paired','ci95_high'],lambda z:z+.01),
    (['long_k','opus_l6',0,'samples',0,'control','wall_ms'],lambda z:z+1),
    (['long_k','opus_l6',0,'speedup_paired','geomean_ratio'],lambda z:z+1),
    (['mma','stress',0,'bit_matches'],lambda z:z-1),
    (['derived','mma','all_dot_products',0],lambda z:z-1),
    (['meta','git_dirty'],lambda _:True),(['meta','code_sha256'],lambda _:'a'*64),
    (['run_config_hash'],lambda _:'a'*16),(['meta','environment','capability'],lambda _:[9,0]),
    (['mma','stress',0,'hardware_sha256'],lambda _:'a'*64),
    (['meta','code_modules'],lambda z:z+['bogus.module']),
    (['zero_regression','legacy_blobs','kernels_blackwell'],lambda _:'a'*40),
    (['timing','cells',0,'dtype'],lambda _:'fp64'),
    (['long_k','opus_l6',0,'shape'],lambda _:[2,16]),
    (['long_k','grid',0,'shape'],lambda _:[0,0])]:
    probe(str(path),edit(path,fn))
def forged(x):
    x['meta']['code_sha256']='0'*64;x['meta']['git_sha']='0'*40
    x['run_config_hash']=p.run_config_hash(False,x['protocol'],'0'*64,'0'*40,x['meta']['environment'])
def falsemodel(x):
    z=x['mma']['stress'][0];z['bit_matches']-=1;z['mismatches']=[{'index':0}];x['derived']=p.derive(x)
def removed(x):
    x['long_k']['opus_l6'].pop();x['derived']=p.derive(x)
def pair_scale(x):
    row=x['timing']['cells'][0]['rounds'][idx]
    for v in ('native','control'): row[v]['gpu_ms']*=1.0001
for name,fn in [('rehash false commit and source',forged),('record a model failure',falsemodel),('remove an Opus cell',removed),('scale a single nonmedian pair',pair_scale)]:probe(name,fn)
elsewhere=T/'checkout'/'results'/f.name
alias=T/'alias.json';alias.symlink_to(elsewhere)
for quick,path in [(True,f),(True,elsewhere),(True,alias),(True,T/'cycle5_c5_01_new.json'),(False,T/'custom.json'),(False,elsewhere),(False,alias)]:
    try:
        with contextlib.redirect_stderr(io.StringIO()): p.parse_args((['--quick'] if quick else [])+['--output',str(path)])
    except SystemExit: print('CLI reject',quick,path.name);continue
    print('CLI accept',quick,path.name)
g=T/'guardrepo';g.mkdir()
def local(*args): subprocess.run(['git','-C',str(g),*args],check=True,stdout=subprocess.DEVNULL)
local('init','-q');(g/'file').write_text('a');local('add','file');local('-c','user.name=Audit','-c','user.email=a@invalid','commit','-qm','fixture')
for state in ['clean','untracked','modified','sm89','cpu']:
    if state=='untracked': (g/'untracked').write_text('new')
    if state=='modified': (g/'file').write_text('dirty')
    if state=='sm89': local('restore','file')
    with patch.object(p.h,'ROOT',g),patch.object(p.torch.cuda,'get_device_capability',return_value=(8,9) if state=='sm89' else (12,0)):
        device=p.torch.device('cpu' if state=='cpu' else 'cuda')
        try:p.require_canonical_host(device,p.h.git_info(),elsewhere,True)
        except RuntimeError as e:print('host reject',state,e);continue
        print('host accept',state)
```

### totals.py

```python
import json,math,statistics as st
from pathlib import Path
d=json.loads(Path('results/cycle5_c5_01_syrk_and_mma.json').read_bytes())
z,t,l,m=[d[k] for k in ['zero_regression','timing','long_k','mma']]
def col(rows,k): return [r[k] for r in rows]
def tally(rows,k): return [sum(col(rows,k)),len(rows)]
def pairs(rows,key='bit_matches'): return [sum(col(rows,key)),sum(col(rows,'cases'))]
def span(xs): return dict(min=min(xs),max=max(xs))
a=z['launches'];zr=dict(cells=len(z['rows']),bitwise_equal=sum(col(z['rows'],'bitwise_equal')),launches_equal=sum(r['native']==r['legacy'] for r in a),launch_cells=len(a),launches_at_512x2048={r['kernel']:r['native'] for r in a if r['shape']==[512,2048]},sync_free=sum(col(a,'native_sync_free')))
tm=dict(cells=len(t['cells']),outputs_bitwise_equal=sum(col(t['cells'],'outputs_bitwise_equal')))
for mode in ['graph','eager']:
    for clock in ['gpu','wall']:
        rs=[c['rounds'] for c in t['cells'] if c['mode']==mode]
        tm[mode+'_'+clock+'_ratio']=span([st.median(r['native'][clock+'_ms'] for r in rows)/st.median(r['control'][clock+'_ms'] for r in rows) for rows in rs])
s,b,i,g,o=[l[k] for k in ['syrk','boundaries','inputs','graphs','opus_l6']]
cells=[(r,v) for r in l['grid'] for v in r['cells'].values()]
rat=[([r['control']['wall_ms'] for r in v['samples']],[r['native']['wall_ms'] for r in v['samples']]) for v in o]
diff=[v['sigma_max']-v['design_ceiling'] for _,v in cells]
lk=dict(syrk_symmetric=tally(s,'symmetric'),syrk_sync_free=tally(s,'sync_free'),syrk_diag_bias_max=max(col(s,'diag_bias_max')),
        boundaries_bitwise_reference=tally(b,'bitwise_equal_reference'),boundaries_unchunked_equal_control=tally([r for r in b if not r['chunked']],'bitwise_equal_control'),
        inputs_bitwise_c4_03=tally(i,'bitwise_equal_c4_03'),inputs_sync_free=tally([r for r in i if r['sync_free'] is not None],'sync_free'),
        graphs_bitwise=tally(g,'bitwise_equal'),graphs_max_abs_diff=max(col(g,'max_abs_diff')),opus_l6_bitwise_c4_03=tally(o,'bitwise_equal_c4_03'),
        opus_l6_speedup=span([st.median(a)/st.median(b) for a,b in rat]),opus_l6_speedup_geomean=span([st.geometric_mean(x/y for x,y in zip(a,b)) for a,b in rat]),
        opus_l6_native_sigma_max=max(r['native']['sigma_max'] for r in o),opus_l6_control_sigma_max=max(r['control']['sigma_max'] for r in o),
        opus_l6_native_above_design=sum(r['native']['sigma_max']>r['design_ceiling'] for r in o),grid_cells=len(cells),grid_chunked_cells=sum(r['chunked'] for r,_ in cells),
        grid_bitwise_c4_03=sum(v['bitwise_equal_c4_03'] for _,v in cells),grid_above_design_ceiling=sum(x>0 for x in diff),grid_above_common_ceiling=sum(v['sigma_max']>d['protocol']['long_k']['ceiling'] for _,v in cells),
        grid_max_excess_over_design=max(diff),grid_sigma_max_worst=max(v['sigma_max'] for _,v in cells))
w,e,stress,sy,au=[m[k] for k in ['witnesses','edge_cases','stress','syrk_configs','audit_seeds']]
allrows=stress+sy+au
mm=dict(witnesses=[sum(float.fromhex(r['sm120_model'])==float.fromhex(r['probe_hardware']) and all(float.fromhex(v)==float.fromhex(r['probe_hardware']) for v in r['syrk_configs_hardware']) for r in w),len(w)],
        witness_permutations=[sum(col(w,'permutations_equal_hardware')),sum(col(w,'permutations'))],exponent_table=tally(m['exponent_table'],'model_equals_hardware'),
        edge_cases=[sum(r['model_equals_hardware'] and r['vectorized_equals_hardware'] for r in e),len(e)],stress_single=pairs([r for r in stress if r['kind']=='single']),
        stress_chains=pairs([r for r in stress if r['kind']=='chain']),scalar_crosscheck=list(map(sum,zip(*(v for v in col(stress,'scalar_crosscheck') if v)))),
        syrk_configs=pairs(sy),syrk_fused_store=pairs(sy,'fused_store_matches'),audit_seeds=pairs(au),audit_seed719_fused_store=pairs([r for r in au if 'fused_store_matches' in r],'fused_store_matches'),
        all_dot_products=pairs(allrows),mismatch_records=sum(map(len,col(allrows,'mismatches'))),hypotheses={r['hypothesis']+'/'+r['dtype']:[r['bit_matches'],r['cases'],r['witness_equals_hardware']] for r in m['hypotheses']},
        ptx=sorted(set(op for group in list(m['ptx_mma'].values())+col(sy,'ptx_mma') for op in group)))
def same(a,b):
    if isinstance(b,dict):assert a.keys()==b.keys();return sum(same(a[k],v) for k,v in b.items())
    if isinstance(b,list):assert len(a)==len(b);return sum(same(x,y) for x,y in zip(a,b))
    if isinstance(b,float):assert abs(a-b)<=1e-12,(a,b)
    else:assert a==b,(a,b)
    return 1
print('derived leaves',same(dict(zero_regression=zr,timing=tm,long_k=lk,mma=mm),d['derived']))
```

### mma.py

```python
import collections, functools, itertools, json, os, pathlib, struct
import numpy as np
import torch
from research.polar_opt import blackwell_mma_model as mm, kernels_blackwell as kb

# Decode to signed integer significand, its unit exponent, and stored field exponent.
@functools.lru_cache(None)
def dec(u,p,eb):
    bias=(1<<(eb-1))-1; mask=(1<<p)-1; f=u&mask; e=(u>>p)&((1<<eb)-1)
    sign=-1 if u>>(eb+p) else 1
    if e==(1<<eb)-1: return ('nan' if f else 'inf',sign,0)
    field=max(1,e)-bias
    return (sign*(f+((1<<p) if e else 0)),field-p,field)
def shift(m,d): return m<<d if d>=0 else m>>-d
def trunc(m,d): return (-1 if m<0 else 1)*shift(abs(m),d)
def oracle(a,b,c,p,eb):
    for s in range(0,len(a),16):
        inf=set(); ts=[]
        def push(m,u,e):
            if m: ts.append((m,u,e))
        x,u,e=dec(c,23,8)
        if x=='nan':return 0x7fffffff
        if x=='inf':inf.add(u)
        else:push(x,u,e)
        for i,j in zip(a[s:s+16],b[s:s+16]):
            x,u,e=dec(int(i),p,eb);y,v,f=dec(int(j),p,eb)
            if x=='nan' or y=='nan' or (x=='inf' and y==0) or (y=='inf' and x==0):return 0x7fffffff
            if x=='inf' or y=='inf':
                inf.add((u if x=='inf' else 1 if x>0 else -1)*(v if y=='inf' else 1 if y>0 else -1))
            else: push(x*y,u+v,e+f)
        if len(inf)>1:return 0x7fffffff
        if inf:c=0x7f800000 if 1 in inf else 0xff800000;continue
        if not ts:c=0;continue
        align=max(e for m,u,e in ts)-25
        z=sum(trunc(m,u-align) for m,u,_ in ts)
        if not z:c=0;continue
        sign=(1<<31) if z<0 else 0; z=abs(z)
        e=z.bit_length()-1+align
        if e>=128:c=sign|0x7f800000;continue
        if e< -126:
            fraction=shift(z,align+149)
            c=(sign|fraction) if fraction else 0
            continue
        c=sign|((e+127)<<23)|(shift(z,align-e+23)-(1<<23))
    return c

def cf(u):return struct.unpack('<f',struct.pack('<I',u))[0]
def cb(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def operand(u,dt):
    return torch.from_numpy(np.ascontiguousarray(u,dtype=np.uint16)).view(dt)
def live(a,b,c,dt,bk):
    # Keep all bit patterns intact up to the load. In particular, never widen sNaNs.
    n,k=a.shape; pad=-n%16
    A=operand(np.pad(a,((0,pad),(0,0))),dt).reshape(-1,16,k)
    B=operand(np.pad(b,((0,pad),(0,0))),dt).reshape(-1,16,k)
    C=np.zeros((len(A),16,16),dtype=np.uint32)
    i=np.arange(n);C[i//16,i%16,i%16]=c
    val,ops=mm.probe_tiles(A,B.mT,torch.from_numpy(C).view(torch.float32),bk=bk,num_warps=1 if bk==16 else 4)
    return val.view(torch.int32).diagonal(dim1=1,dim2=2).flatten()[:n].cpu().numpy().view(np.uint32),ops
def hx(a): return ' '.join(f'{int(v):04x}' for v in a)
def sweep(label,a,b,c,bk=16):
    a=np.asarray(a,dtype=np.uint16);b=np.asarray(b,dtype=np.uint16);c=np.asarray(c,dtype=np.uint32)
    hw,ops=live(a,b,c,dt,bk); assert ops==[op],ops
    counts=collections.Counter()
    av=operand(a,dt).double().tolist();bv=operand(b,dt).double().tolist()
    vec=mm.sm120_mma_batched(operand(a,dt),operand(b,dt),torch.from_numpy(c).view(torch.float32),dtype=name).view(torch.int32).numpy().view(np.uint32)
    for i,(aa,bb,cc,out) in enumerate(zip(a,b,c,hw)):
        rule=oracle(aa,bb,int(cc),p,eb)
        scalar=cb(mm.sm120_mma_accumulate(av[i],bv[i],cf(int(cc)),dtype=name))
        v=int(vec[i]);out=int(out)
        for key,x in [('rule',rule),('scalar',scalar),('vector',v)]: counts[key]+=x==out
        if out!=rule or out!=scalar or out!=v:
            record={'dtype':name,'set':label,'index':i,'a':hx(aa),'b':hx(bb),'c':f'{cc:08x}','oracle':f'{rule:08x}','scalar':f'{scalar:08x}','vector':f'{v:08x}','hw':f'{out:08x}'}
            dest.write(json.dumps(record)+'\n')
            if i==0 or (len(a)<100 and out!=rule): print('W',record)
        if len(a)==1:print('BITS',name,label,f'{out:08x}',f'{rule:08x}',f'{scalar:08x}',f'{v:08x}')
    print('M',name,label,len(c),*[counts[k] for k in ('rule','scalar','vector')])
def raw(vals):return torch.tensor(vals,dtype=dt).view(torch.uint16).tolist()
def val(v):return raw([v])[0]
dest=open(pathlib.Path(os.environ['TMPDIR'])/'witness.jsonl','w')
torch.set_num_threads(1)
for j,(name,dt,p,eb) in enumerate([('fp16',torch.float16,10,5),('bf16',torch.bfloat16,7,8)]):
    tok=('f16','bf16')[j]; op='mma.sync.aligned.m16n8k16.row.col.f32.'+tok+'.'+tok+'.f32'
    print('PTX expected',op)
    rng=np.random.default_rng(619731+j)
    def U(lo,hi,shape):return rng.integers(lo,hi,size=shape,dtype=np.uint16)
    def signs(shape):return U(0,2,shape)<<15
    m=(1<<p)-1;inf=((1<<eb)-1)<<p
    # One row per named witness. Expected NaN bits come directly from the oracle.
    cases=[]
    def case(label,a,b,c=0): cases.append((label,a,b,c))
    one=val(1.);neg=val(-1.);largest=inf-1
    case('negzero',[0],[0],0x80000000);case('cancel',[one,neg],[one,one],0x80000000)
    case('zero_negative_product',[0x8000],[0]);case('carry',[one+m]*16,[one+m]*16,cb(1.999))
    case('carry_cancel',[one+m]*16+[(one+m)|0x8000]*16,[one+m]*32,0)
    case('maxacc',[0],[0],0x7f7fffff);case('maxacc_neg',[0],[0],0xff7fffff)
    case('max16',[largest]*16,[one if j else largest]*16)
    case('infx0',[inf],[0]);case('mixedinf',[inf,inf|0x8000],[one,one])
    case('neg_inf',[inf|0x8000],[one]);case('pos_inf',[inf],[one])
    for sn in (1,1<<(p-1)):
        case('nan_a'+str(sn),[inf|sn],[one]);case('nan_b'+str(sn),[one],[inf|sn])
    case('snan_c',[0],[0],0x7f800001);case('qnan_c',[0],[0],0x7fc00000)
    if j:
        case('overflow_then_cancel',raw([2.**70,-2.**70,1]),raw([2.**70,2.**70,1]),0x7f7fffff)
        case('negative_tiny',raw([-2.**-80]),raw([2.**-80]))
        case('minnormal_from_below',raw([2.**-75]),raw([2.**-74]),0x007fffff)
        case('subnormal_from_above',raw([-2.**-75]),raw([2.**-74]),0x00800000)
    w=mm.BF16_WITNESS if j else mm.ASTRA_FP16_WITNESS
    row=[val(w.get(i,0.)) for i in range(16)];case('C4_witness',row,row)
    for label,a,b,c in cases:
        k=((len(a)+15)//16)*16
        sweep(label,[a+[0]*(k-len(a))],[b+[0]*(k-len(b))],[c])
    for label in ('normal','full_finite','both_sub','sub_x_normal','fp32_subacc','near_fp32_min'):
        n=1024;shape=(n,16); C=rng.integers(0,0x00800000,n,dtype=np.uint32)|(rng.integers(0,2,n,dtype=np.uint32)<<31)
        a=U(0,inf,shape);b=U(0,inf,shape)
        if label=='normal':
            a=val(2.**-12)+U(0,10*(m+1),shape);b=val(2.**-12)+U(0,10*(m+1),shape)
            C=rng.uniform(-1,1,n).astype('f').view('I')
        if label=='full_finite':C=rng.integers(0,0x7f800000,n,dtype=np.uint32)|0x80000000
        if label in ('both_sub','sub_x_normal'):a=U(1,m+1,shape)
        if label=='both_sub':b=U(1,m+1,shape)
        if label=='near_fp32_min' and j:
            a=val(2.**-70)+U(0,9*(m+1),shape);b=val(2.**-66)+U(0,9*(m+1),shape)
        sweep(label,a|signs(shape),b|signs(shape),C)
    for k in (16,32,64,128,256,512,1024,2048,4096):
        shape=(32,k)
        a=val(2.**-10)+U(0,10*(m+1),shape);b=val(2.**-10)+U(0,10*(m+1),shape)
        sweep('chain'+str(k),a|signs(shape),b|signs(shape),rng.uniform(-1,0,32).astype('f').view('I'),min(128,k))
    shape=(9,257)
    X=val(2.**-10)+U(0,10*(m+1),shape)|signs(shape)
    for cfg in kb._CONFIGS:
        O,ops=mm.syrk_raw(operand(X,dt).cuda()[None],mm.config_record(cfg))
        F,ops2=mm.syrk_raw(operand(X,dt).cuda()[None],mm.config_record(cfg),dt)
        assert ops==ops2==[op]
        oracle_bits=np.array([[oracle(x,y,0,p,eb) for y in X] for x in X],dtype='I')
        actual=O.cpu().view(torch.uint32).numpy()[0]
        cast=torch.from_numpy(oracle_bits).view(torch.float32).to(dt).view(torch.uint16).numpy()
        fused=F.cpu().view(torch.uint16).numpy()[0]
        print('CFG',name,mm.config_record(cfg),np.sum(actual==oracle_bits),np.sum(fused==cast),actual.size)
        assert np.array_equal(actual,oracle_bits) and np.array_equal(fused,cast)
dest.close()
```

### extreme.py

```python
# Share only the independent integer oracle and the raw-bit hardware adapter.
from pathlib import Path
import os
src=(Path(os.environ['TMPDIR'])/'mma.py').read_text().split('dest=open(')[0]
exec(src)
torch.set_num_threads(1)
dt=torch.bfloat16;name='bf16';p=7;eb=8;op='mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32'
dest=open(Path(os.environ['TMPDIR'])/'extreme-witness.jsonl','w')
rng=np.random.default_rng(173901)
def fields(es):
    return ((es+127).astype('H')<<7)|rng.integers(0,128,es.shape,dtype='H')|(rng.integers(0,2,es.shape,dtype='H')<<15)
for label,lo,hi in [('underflow',-170,-153),('sub_boundary',-153,-125),('overflow',118,146),('cancellation',-115,125)]:
    n=512; shape=(n,16)
    E=rng.integers(lo,hi,size=(n,1)); exp=E+rng.integers(-6,3,shape)
    ax=exp//2+rng.integers(-12,12,shape);bx=exp-ax;a=fields(ax);b=fields(bx)
    C=np.zeros(n,dtype='I')
    if label=='sub_boundary':C[::2]=rng.integers(0,0x00800001,n//2,dtype='I')|0x80000000
    if label=='overflow':C[::2]=0xff7fffff
    if label=='cancellation':
        a[:,8:]=a[:,:8]^0x8000;b[:,8:]=b[:,:8];b[:,15]^=1
    sweep(label,a,b,C)
dest.close()
```

### historical.py

```python
import os
from pathlib import Path
exec((Path(os.environ['TMPDIR'])/'mma.py').read_text().split('dest=open(')[0])
for dt,name,p,eb,w in [(torch.float16,'fp16',10,5,mm.ASTRA_FP16_WITNESS),(torch.bfloat16,'bf16',7,8,mm.BF16_WITNESS)]:
    x=torch.tensor([w.get(i,0) for i in range(16)],dtype=dt)
    a=x.view(torch.uint16).numpy()
    independent=oracle(a,a,0,p,eb)
    scalar=cb(mm.sm120_mma_accumulate(x.tolist(),x.tolist(),dtype=name))
    h=[]
    for cfg in kb._CONFIGS:
        out,_=mm.syrk_raw(x.cuda().reshape(1,1,16),mm.config_record(cfg))
        h.append(out.view(torch.uint32).item())
    assert h==[independent]*11==[scalar]*11
    print('HIST',name,hx(a),'C=00000000',hex(independent),'11/11')
```

### devicebits.py

```python
import collections,json,os
from pathlib import Path
import numpy as np,torch
from research.polar_opt import blackwell_mma_model as mm
t=Path(os.environ['TMPDIR'])
rows=[json.loads(l) for f in ['witness.jsonl','extreme-witness.jsonl'] for l in (t/f).read_text().splitlines()]
for name,dt in mm.DTYPES.items():
    group=[r for r in rows if r['dtype']==name]
    operands=[torch.from_numpy(np.array([[int(w,16) for w in r[k].split()] for r in group],dtype='H')).view(dt).cuda() for k in ['a','b']]
    acc=torch.from_numpy(np.array([int(r['c'],16) for r in group],dtype='I')).view(torch.float32).cuda()
    v=mm.sm120_mma_batched(*operands,acc,dtype=name).view(torch.int32).cpu().numpy().view('I')
    print('GPU-vector',name,dict(collections.Counter((r['hw'],r['vector'],f'{x:08x}') for r,x in zip(group,v))))
```

### gridcontrol.py

```python
import ctypes, torch
from research.polar_opt import kernels_blackwell as kb
drv=ctypes.CDLL('libcuda.so.1');drv.cuInit(0);limit=ctypes.c_int();assert drv.cuDeviceGetAttribute(ctypes.byref(limit),6,0)==0
print('grid.y',limit.value)
# A contiguous input also fails. Pin one legal production config to isolate the launch geometry.
x=torch.ones(1,1,1<<28,device='cuda',dtype=torch.float16);o=torch.empty(1,1,1,device='cuda')
def launch(kernel,grid,piece,out,n,k,*strides):
    cfg=next(c for c in kb._CONFIGS if c.kwargs['BLOCK']==32)
    kernel.fn[grid(cfg.kwargs)](piece,out,n,k,*strides,**cfg.kwargs,num_warps=cfg.num_warps,num_stages=cfg.num_stages)
for count in [65535,65536]:
    try:kb._syrk_chunked(x,o,launch_fn=launch,max_bytes=4*count);torch.cuda.synchronize(); print(count,o.item(),int(o.item())==1<<28)
    except RuntimeError as e: print(count,type(e).__name__,e)
try:
    y=kb.triton_syrk(x[0],torch.float32)
    torch.cuda.synchronize()
    print('public contiguous',y.item())
except RuntimeError as e: print('public contiguous',type(e).__name__,e)
```

### hook.py

```python
import torch
from research.polar_opt import bench_c5_01_syrk_and_mma as c
torch.manual_seed(983)
x=torch.randn(1,3,100000,device='cuda').half();out=torch.empty(1,3,3,device='cuda',dtype=torch.float32)
def read():
    c.psb._gram(x,out);return out.clone()
with c.ungated():single=read()
native=read()
for gate in [65536,131072]:
    h=c.c3.ChunkedGram(4096,gate)
    with h.patched(): val=read()
    print('hook',gate,h.launches,'equal_native',torch.equal(val,native),'equal_single',torch.equal(val,single),'entry',val[0,0,0].item(),'single',single[0,0,0].item())
print('restored',c.ssb.SYRK_CHUNK_GATE,c.psb.SYRK_CHUNK_GATE)
```

### checks.py

```python
import ctypes, hashlib, json, os, subprocess
import torch
from research.polar_opt import bench_c5_01_syrk_and_mma as c,kernels_blackwell as kb
from pathlib import Path
d=json.loads(c.CANONICAL_OUTPUT.read_text());m=d['meta'];keys=('torch','triton','cuda','driver','device','capability')
conf=dict(schema=d['schema_version'],quick=False,protocol=d['protocol'],code_sha256=m['code_sha256'],git_sha=m['git_sha'],env={k:m['environment'][k] for k in keys})
print('config',hashlib.sha256(json.dumps(conf,sort_keys=True).encode()).hexdigest()[:16],d['run_config_hash'])
for name,oid in d['zero_regression']['legacy_blobs'].items():
    assert subprocess.check_output(['git','rev-parse',c.LEGACY_BASE+':research/polar_opt/'+name+'.py'],text=True).strip()==oid
print('legacy',d['zero_regression']['legacy_blobs'])
z=torch.ones(1,1,device='cuda',dtype=torch.float16);o=torch.empty(1,1,1,device='cuda')
cfg=dict(BLOCK=32,BK=128,num_warps=4,num_stages=3)
for stride in (2**24-1,2**24,2**31-1,2**31,2**32):
    v=z.as_strided((1,1,1),(1,1,stride))
    try:kb._syrk_chunked(v,o)
    except ValueError as e:print('stride guard',stride,'rejected');continue
    J=kb._syrk_kernel.fn.warmup(z,o,1,1,1,1,stride,**cfg,grid=(1,1))
    sig=next(l for l in J.asm['ttir'].splitlines() if 'tt.func ' in l)
    print('stride guard',stride,'accepted',o.item(),sig)
x=torch.randn(2,3,70000,device='cuda')
for name,fn in [('syrk',lambda:kb.triton_syrk(x.half())),('PS',lambda:c.psb.polar_switch_blackwell(x)),('SS',lambda:c.ssb.schatten_switch_blackwell(x)),('SS_frob',lambda:c.ssb.schatten_switch_blackwell(x,scaling='frobenius')),('SS_sigma',lambda:c.ssb.schatten_switch_blackwell(x,sigma_hat=.9))]:
    fn();torch.cuda.synchronize();torch.cuda.set_sync_debug_mode('error')
    try:fn();sync='pass'
    except RuntimeError as e: sync=str(e)
    finally:torch.cuda.set_sync_debug_mode('default')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:fn()
    transfers=[e.name for e in prof.events() if 'memcpy' in e.name.lower() or 'dtoh' in e.name.lower()]
    print('sync',name,sync,'copies',transfers)
```

### capture.py

```python
import torch
from research.polar_opt import bench_c5_01_syrk_and_mma as c
x=torch.randn(512,2048,device='cuda',generator=torch.Generator(device='cuda').manual_seed(7149))
f=c.circuit('ps3_55','fp16');e=f(x)
gs=[c._graph(f,x,v) for v in c.VARIANTS]; torch.cuda.synchronize()
def cmp():
    a,b=[o for g,o in gs];return c.bit_equal(a,b),c.bit_equal(a,e),c.bit_equal(b,e)
print('production pre',cmp())
for g,_ in gs:g.replay()
print('production post',cmp())
# Fixed sentinel buffers make a false positive deterministic.
buf={v:torch.zeros(1,device='cuda') for v in c.VARIANTS}
def candidate(z):
    v='native' if c.kbw.SYRK_CHUNK_GATE==65536 else 'control'
    if torch.cuda.is_current_stream_capturing(): buf[v].fill_(1 if v=='native' else 2)
    return buf[v]
gs=[c._graph(candidate,x,v) for v in c.VARIANTS]
print('distinct graphs pre',[o.item() for _,o in gs])
for g,_ in gs:g.replay()
print('distinct graphs post',[o.item() for _,o in gs])
```

### syrk.py

```python
import contextlib, math, torch
from research.polar_opt import kernels_blackwell as kb, bench_c5_01_syrk_and_mma as c, bench_c4_03_wide_and_gram_drift as old
torch.manual_seed(631921)
device='cuda'
def bits(t): return t.contiguous().view({torch.float16:torch.int16,torch.bfloat16:torch.int16,torch.float32:torch.int32}[t.dtype])
def eq(a,b):return torch.equal(bits(a),bits(b))
def check_sync(fn):
    fn();torch.cuda.synchronize()
    mode=torch.cuda.get_sync_debug_mode();torch.cuda.set_sync_debug_mode('error')
    try:fn()
    finally:torch.cuda.set_sync_debug_mode(mode)
def gram_test(n,k,b,layout,cap=None):
    raw=torch.zeros((b,2*k+1,n) if layout=='tall' else (b,n,2*k+1),device=device,dtype=torch.float16)
    v=raw[:,1::2,:].mT if layout=='tall' else raw[:,:,1::2]
    cols=sorted({0,15,4095,4096,k-1,*range(8192,k,4096)})
    coeff=torch.randint(-3,4,(b,n,len(cols)),device=device).half()
    v[:,:,cols]=coeff
    exact=coeff.double()@coeff.double().mT
    for dt in (torch.float16,torch.bfloat16,torch.float32):
        out=kb.triton_syrk(v,dt);assert eq(out,exact.to(dt)) and eq(out,out.mT)
        if cap is not None:
            launches=[]
            def launch(kernel,grid,piece,*args):
                launches.append(piece.shape[0]);kernel[grid](piece,*args)
            alt=torch.empty_like(out);kb._syrk_chunked(v,alt,launch_fn=launch,max_bytes=cap)
            assert eq(out,alt);print('batches',n,k,dt,launches)
    print('gram',n,k,b,layout,'3/3')
for k in (65535,65536,65537,69632,69633,100000):
    for layout in ('tall','wide'):gram_test(3,k,2,layout)
gram_test(1,70000,2,'wide');gram_test(512,70000,1,'wide',4*512**2*4)
# Production cap 256 MiB, 16 chunks per launch for n=2048, count=17.
gram_test(2048,70000,1,'tall')
for dt,p,unit in [(torch.float16,10,0x3c00),(torch.bfloat16,7,0x3f80),(torch.float32,23,0x3f800000)]:
    # One product per chunk; no partial-rounding ambiguity. Two equal squares if p+1 is odd.
    exp=p+1; base=2.**-((exp+1)//2); terms=[1]+[base]*(2 if exp%2 else 1)+[2.**(-24 if p==23 else -13)]
    x=torch.zeros(2,65537,dtype=torch.float16,device=device)
    for i,v in enumerate(terms): x[:,i*4096]=v
    x[1,:].neg_()
    out=kb.triton_syrk(x,dt)
    vals=[int(q)&(0xffffffff if p==23 else 0xffff) for q in bits(out).flatten().tolist()]
    expected=[unit+1,(unit+1)|(1<<(31 if p==23 else 15)),(unit+1)|(1<<(31 if p==23 else 15)),unit+1]
    assert eq(out,out.mT)
    print('round',dt,terms, 'actual',*[hex(i) for i in vals],'single',*[hex(i) for i in expected])
def graph(fn,x):
    side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):fn(x);fn(x)
    torch.cuda.current_stream().wait_stream(side)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): y=fn(x)
    saved=[]
    for i in range(3):
        x.normal_();g.replay();assert eq(y,fn(x));saved.append(y.clone())
    assert not eq(saved[0],saved[1]);check_sync(lambda:fn(x))
for name in ('ps3_55','ss3_54','ss3_53'):
    for dtype in ('fp16','bf16'):
        f=c.circuit(name,dtype)
        for k in (65535,65536,65537,69632,69633,100000):
            for tall in (True,False):
                z=torch.randn((2,3,2*k),device=device)[:,:,::2]
                z=z.mT if tall else z
                y=f(z)
                with old.ChunkedGram().patched():assert eq(y,f(z))
        for shape in [(1,70000),(70000,1)]:
            z=torch.randn(shape,device=device);y=f(z);assert y.shape==z.shape and y.isfinite().all() and y.abs().sum()>0
        x=torch.randn(5,3,70000,device=device);x[1]=0;x[2]*=1e-20;x[3,0,0]=float('nan');x[4,0,0]=float('inf')
        y=f(x);print('mixed',name,dtype,[bool(v.isfinite().all()) for v in y],[bool((v==0).all()) for v in y]);assert eq(y[0],f(x[0])) or torch.allclose(y[0],f(x[0]),atol=1e-4,rtol=1e-3)
        if name.startswith('ss'):
            assert (y[1:]==0).all()
            for opts in ({'sigma_hat':0.99},{'scaling':'frobenius'}):
                ff=lambda z:c.ssb.schatten_switch_blackwell(z,name.upper(),dtype=c.mm.DTYPES[dtype],allow_unsafe_bf16=True,**opts)
                v=ff(x[:2])
                with old.ChunkedGram().patched():assert eq(v,ff(x[:2]))
                if 'sigma_hat' not in opts:check_sync(lambda:ff(x[:2]))
        x=torch.randn(2,3,70000,device=device);graph(f,x)
        print('circuit',name,dtype,'12 boundaries; 2 min==1; 3 graph overwrites; warm sync OK')
v=torch.ones(1,2,2**25,dtype=torch.float16,device=device)[:,:,::2**24]
try:kb._syrk_chunked(v,torch.empty(1,2,2,device=device))
except ValueError as e:print('stride',v.stride(),e)
else:raise AssertionError('guard missing')
graph(lambda z:kb.triton_syrk(z.half(),torch.float32),torch.randn(2,1,70000,device=device))
```

### large.py

```python
import json, math, numpy as np, torch
from research.polar_opt import bench_c5_01_syrk_and_mma as c
c.set_numerics(torch.device('cuda'))
artifact=json.loads(c.CANONICAL_OUTPUT.read_bytes())
# FP64 block Grams on GPU, eigendecomposition in CPU NumPy; work with short matrices only.
def gram(v,w):
    pieces=[]
    for x,y in zip(v.split(349525,dim=1),w.split(349525,dim=1)):
        pieces.append((x.double()@y.double().T).cpu().numpy())
    return np.sum(pieces,axis=0)
for shape in [(4,1<<25),(1<<25,2),(1,1<<25)]:
    g=torch.randn(shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(0))
    v=g if shape[0]<=shape[1] else g.T
    ev,U=np.linalg.eigh(gram(v,v));invsqrt=(U/np.sqrt(ev))@U.T
    print('G',shape,ev.tolist())
    for r in artifact['long_k']['opus_l6']:
        if list(shape)!=r['shape']:continue
        q=c.circuit(r['kernel'],r['dtype'])(g);q=q if shape[0]<=shape[1] else q.T
        qq=gram(q,q);sigma=np.sqrt(np.linalg.eigvalsh(qq));cross=gram(q,v)
        err=math.sqrt(max(0,np.trace(qq)+len(ev)-2*np.trace(cross@invsqrt))/len(ev))
        ref=r['native']; delta=max(abs(err-ref['rel_fro_err']),abs(sigma[0]-ref['sigma_min']),abs(sigma[-1]-ref['sigma_max']))
        print('L6',shape,r['kernel'],r['dtype'],sigma.tolist(),err,'delta',delta,'ceiling',r['design_ceiling'])
        assert sigma[-1]<r['design_ceiling'] and np.isfinite(err)
        del q
    del g,v
    torch.cuda.empty_cache()
```

### latency.py

```python
import contextlib, functools, statistics, tempfile, time, os
from pathlib import Path
import torch, numpy as np
from scipy.stats import t
from research.polar_opt import bench_c5_01_syrk_and_mma as c
c._private_tmp=lambda:Path(tempfile.mkdtemp(dir=os.environ['TMPDIR']))
c.set_numerics(torch.device('cuda'))
old=c.load_legacy()
dev='cuda';torch.manual_seed(473019)
flush=torch.empty(4*torch.cuda.get_device_properties(0).L2_cache_size,device=dev,dtype=torch.uint8)
def graph(fn):
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): y=fn()
    return g,y
def sample(f):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    flush.zero_();torch.cuda.synchronize();now=time.perf_counter_ns()
    a.record();f();b.record();torch.cuda.synchronize()
    return a.elapsed_time(b), (time.perf_counter_ns()-now)*1e-6
for shape in [(512,2048),(2048,8192)]:
    x=torch.randn(shape,device=dev)
    for nm in c.KERNELS:
        native=functools.partial(c.circuit(nm,'fp16'),x)
        ctrl=functools.partial(c.circuit(nm,'fp16',old['modules']),x)
        y=native();c.share_autotune(old);z=ctrl();assert torch.equal(y.view(torch.int32),z.view(torch.int32))
        gs=[graph(fn) for fn in (ctrl,native)]
        for captured,_ in gs:captured.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(out.view(torch.int32),y.view(torch.int32)) for _,out in gs)
        for mode,fns in [('graph',[g.replay for g,y in gs]),('eager',[ctrl,native])]:
            for f in fns: sample(f); sample(f)
            raw=[]
            for i in range(24):
                row=[None,None]
                for j in (i%2,1-i%2):row[j]=sample(fns[j])
                raw.append(row)
            a=np.array(raw); med=np.median(a,axis=0);r=med[1]/med[0]
            lr=np.log(a[:,1]/a[:,0]);half=t.ppf(.975,23)*lr.std(0,ddof=1)/np.sqrt(24)
            print('LIVE',shape,nm,mode,'native_ms',med[1].tolist(),'legacy_ms',med[0].tolist(),'ratio',r.tolist(),'ci',np.exp(lr.mean(0)-half).tolist(),np.exp(lr.mean(0)+half).tolist())
        del gs,y,z
```

### repeat.py

```python
import os
from pathlib import Path
text=(Path(os.environ['TMPDIR'])/'latency.py').read_text()
exec(text.split('for shape in [(512,2048)')[0])
torch.randn((512,2048),device='cuda')
x=torch.randn((2048,8192),device='cuda')
n=functools.partial(c.circuit('ss3_53','fp16'),x);b=functools.partial(c.circuit('ss3_53','fp16',old['modules']),x)
n();c.share_autotune(old);b()
for captures in ((n,b),(b,n)):
    pairs={f:graph(f) for f in captures}
    for f in (n,b): sample(pairs[f][0].replay)
    raw=[]
    for i in range(64):
        row={}
        for f in ((n,b) if i%2==0 else (b,n)):row[f]=sample(pairs[f][0].replay)
        raw.append([row[n],row[b]])
    a=np.array(raw);r=np.median(a[:,0],axis=0)/np.median(a[:,1],axis=0)
    lr=np.log(a[:,0]/a[:,1]);h=t.ppf(.975,63)*lr.std(0,ddof=1)/8
    print('repeat', 'native-first-capture' if captures[0]==n else 'legacy-first-capture',r.tolist(),np.exp(lr.mean(0)-h).tolist(),np.exp(lr.mean(0)+h).tolist())
    del pairs
```

### gridlimit.py

```python
import torch
from research.polar_opt import kernels_blackwell as kb
x=torch.ones(1,1,device='cuda',dtype=torch.float16).expand(1,4096*65536)
try:
    y=kb.triton_syrk(x,torch.float32);torch.cuda.synchronize()
    print('wide_grid_result',y.item())
except Exception as e:
    print('wide_grid_error',type(e).__name__,str(e));raise
```

### harvest.py

```python
import pathlib,pytest
class Data:
    def pytest_collection_modifyitems(self,items):
        for item in items:
            item.module.TRAJECTORIES=pathlib.Path('/home/ryan/Documents/projects/ai/artifacts/harvested_trajectories.pt')
raise SystemExit(pytest.main(['-p','no:cacheprovider','tests/test_schatten_switch.py','-q','-k','tier1_117'],plugins=[Data()]))
```

</details>

## Post-Remediation Live-GPU Re-Audit Addendum

**FAIL. Keep the hold on canonical promotion.** The RTX 5080 reproduced the v2 numerical results, including the alignment floor at −133. The canonical validator still certified a payload in which the recorded model and hardware witnesses differ by one FP32 bit. It also certified a payload after all 903,100 audit-seed results were removed. Both defects can be detected from the JSON and the protocol alone. The committed artifact passed independent replay.

### Findings at the audited revision

| Baseline finding | Result | Evidence |
|---|---|---|
| 1. CUDA grid-y limit | Pass | A contiguous FP16 tensor of shape `(1, 268435456)` filled with ones produced exactly `268435456.0` through both `triton_syrk` and `_syrk_chunked`. Both issued grid-y sizes `[65535, 1]`. The driver reported a limit of 65,535. A 65,540-byte partial-results cap gave `[16385, 16385, 16385, 16381]` and the same sum. |
| 2. FP64 single rounding | Pass | All 12 destination words of the three 2×2 Grams at K = 65,537 matched. The positive/negative pairs were `3c01/bc01` (FP16), `3f81/bf81` (BF16) and `3f800001/bf800001` (FP32). Both off-diagonals matched bitwise. The raw kernel and its wrapper also passed the midpoint sweep below. |
| 3. Raw MMA words | Pass | All five predictions matched all 14,954 original hardware outputs, including the 16 NaNs (`7fffffff`) and the negative-underflow cases (+0). The baseline config sweep matched 1,782/1,782 raw FP32 words and 1,782/1,782 fused words across all 11 configs and both formats. |
| 4. CUDA graph equality | Pass | Starting with equal zero buffers, graphs writing 1 and 2 returned false. Two graphs writing 1 returned false against eager 2 and true against eager 1. The observed order was native replay, control replay, device synchronization, first bit comparison. Production PS3_55 graphs also equaled fresh eager results before and after negating the static input. |
| 5. Canonical validation and CLI | **Partial. Hold remains.** | All six named malformed payloads and the named CLI targets were rejected. Further canonical probes accepted contradictory witness evidence and missing MMA populations. See the reproductions below. |
| 6. Historical custom gate | Pass | At K = 100,000, `ChunkedGram(4096, 131072)` counted zero hook launches on both PS3_55 and SS3_54. Each output equaled the forced single-launch Gram path bitwise and differed from the default chunked path. Both circuit gates returned to 65,536 and both callable identities were restored on normal exit and exception. |
| 7. Williams balance and v2 timings | Pass | All 30 ordinary cells balanced 20/20/20/20 over 40 rounds. All 18 Opus cells balanced 2/2/2/2 over four rounds. All stored timing statistics and all 137 derived scalar leaves agreed within absolute tolerance 1e−12. |
| 8. Minimum MMA alignment exponent | Pass on this RTX 5080 | The independent integer rule with floor −133 matched all 76,788 additional direct outputs. Removing the floor caused 3,362 mismatches. Both adjacent candidate floors also failed. |

### Single-rounding evidence

The reference enumerated adjacent positive destination words as exact rational values. For each pair it tested the lower endpoint, the exact midpoint and its two adjacent binary64 values, then repeated all four with the sign reversed. At a midpoint it selected the even destination word. At overflow it used the next unbounded binade to define the midpoint. This covers all 31,744 FP16 boundaries and 32,640 BF16 boundaries, including the first subnormal, the carry into the minimum normal, and overflow. It exercises both sides of 2⁻²⁵ and 2⁻¹³⁴ at binary64 resolution. Additional input words covered signed zeros, both infinities, extreme finite binary64 values, and signaling and quiet NaNs of both signs.

| Destination | Audited input words | Direct conversion kernel | Store wrapper |
|---|---:|---:|---:|
| FP16 | 253,962 | 253,962 matched | 253,962 matched |
| BF16 | 261,130 | 261,130 matched | 261,130 matched |

The four entry-point/format combinations each passed eager execution and two graph replays with overwritten output buffers. Reversing the source between replays changed the expected output order. Each warmed call succeeded with CUDA sync-debug set to `error`. CPU/CUDA profiler traces recorded zero device-to-host copies in the four measured calls. Comparisons that intentionally copied results to the CPU ran outside those measurements.

### Independent hardware test of the alignment floor

The baseline integer oracle above supplied the raw-word decoder, k16 chain and FP32 encoder. The tested change to its alignment unit was `align = max(floor, max(e for m, u, e in ts)) - 25`. Each hardware result was read through an int32 view. The comparisons used the independent oracle, the scalar result encoded separately by `struct.pack` and PyTorch, and the batched result on CPU and CUDA. No comparator rewrote NaN payloads or zero signs. The compiled instruction was `mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}.{f16,bf16}.f32` with matching operand formats.

| Direct instruction corpus | Hardware outputs | Matches for **each** of the five comparisons | Matches without a floor | Matches at −134 | Matches at −132 |
|---|---:|---:|---:|---:|---:|
| Original FP16 and BF16 cases, including chains through K = 4,096 | 14,954 | 14,954 | 14,954 | 14,954 | 14,953 |
| BF16 random normal operands, stratified by maximum product field exponent | 19,840 | 19,840 | 19,838 | 19,838 | 19,834 |
| BF16 two-lane subtraction and lane permutations | 48,000 | 48,000 | 44,640 | 46,080 | 46,080 |
| BF16 subnormal × normal operands with randomized operand order | 8,928 | 8,928 | 8,928 | 8,928 | 8,926 |
| FP16 minimum-magnitude and zero controls | 20 | 20 | 20 | 20 | 20 |
| **Total** | **91,742** | **91,742** | **88,380** | **89,820** | **89,813** |

The BF16 random sweep used NumPy seed `0xc501133`. It included 128 vectors at each maximum product field-exponent sum from −160 through −130. Each vector ran with C = 0, ±2⁻¹⁴⁹ and ±2⁻¹³³. The subnormal-operand sweep used seed 133158, 48 vectors per exponent and both signs of zero as distinct C inputs. A nonzero subnormal FP32 accumulator contributes field exponent −126 in this rule. The C = 0 rows distinguish the competing floors. FP16's minimum product **field-exponent sum** is −28, and its smallest nonzero exact product is 2⁻⁴⁸. The requested −160 to −130 product range is representable with BF16 operands.

Two lanes suffice to distinguish both neighboring floors. All other lanes and C were +0. The hexadecimal inputs are BF16 words.

| A[0], B[0] | A[1], B[1] | Exact products | Hardware FP32 | Alternative prediction |
|---|---|---|---|---|
| `1a00, 1a80` | `9780, 1800` | 2⁻¹⁴⁹ and −2⁻¹⁵⁹ | `00000001` | Floor −134 gives `00000000`. |
| `1a00, 1a80` | `9800, 1800` | 2⁻¹⁴⁹ and −2⁻¹⁵⁸ | `00000000` | Floor −132 gives `00000001`. |

Together these cases locate the per-term quantum at 2⁻¹⁵⁸ for these operations. The permutation sweep tested all 240 ordered pairs of distinct k16 lanes, both product signs, four leading exponents and five subtractor exponents. A separate 9×33 BF16 Gram containing the boundary pairs matched another **891/891** raw words and **891/891** fused words across all 11 production configs. Config counts include mirrored entries.

### Canonical artifact and validator

The audit checkout started clean at `4057daf01b5db502eab8c53e3f458695e9cc94e7`. Its three direct ancestors were `e75028729ce9e472b9a06a7a3eb1842d0896a187`, `307fe7a8dc4fa7a9ee7642c26e33c42e6ec8293f` and `289e6d75fc341ce6476867147a000da3d4704969`. The artifact commit added only `results/cycle5_c5_01_syrk_and_mma.json`. Both remediation commits passed `git show --check`.

| Check | Independent result |
|---|---|
| Artifact | SHA-256 `52821ce629c8dca05bd22bab08088521a30a02b6ea23a70fb440bc6ec3b80271`. The file equals the committed blob byte for byte. Strict JSON parsing, finite serialization and `validate_results(payload, canonical=True)` passed. |
| Source | The AST import walk over Git blobs at `e750287` found 16 modules. Hashing each sorted module name, NUL and file bytes produced `f82ec9df16e35661bf62e99c88342adcdfda99bd1f94090e5e527dc8b319aed7`. Names and digest matched the JSON. |
| Metadata | `git_sha` equals the full `e750287` commit above, `git_dirty` is false and `schema_version` is `cycle5_c5_01.v2`. All three legacy object IDs matched Git. The recomputed run hash was `4eccd8fbe8b68189`. |
| Hardware digests | Regenerating the producer's canonical inputs and hashing fresh unmodified FP32 device outputs reproduced **226/226** digests. These cover 64,512 stress results in 28 rows, 59,422 SYRK results in 88 rows, and 903,100 audit-seed results in 110 rows. All **1,027,034/1,027,034** matched the batched model. These seeded replays are separate from the independent-oracle table. |

Canonical validation rejected timing dtype `fp64`, a missing Opus row after re-derivation, Opus shape `[2,16]`, grid shape `[0,0]`, a reported stress mismatch after re-derivation, and a bogus entry in `code_modules`. Canonical CLI parsing rejected the canonical basename in another checkout, file and directory symlinks leading to the remediation checkout's artifact, and a file symlink to the local artifact. The baseline quick-mode reserved-name checks also rejected their targets.

**Open part of finding 5.** Two minimal, independent mutations of the valid payload still passed canonical validation. The first changes one FP32 bit in a stored numeric witness. The second drops the entire audit-seed population. Run each against a fresh parse of the committed JSON.

```python
import json
from research.polar_opt import bench_c5_01_syrk_and_mma as c5

p = json.loads(c5.CANONICAL_OUTPUT.read_bytes())
w = p["mma"]["witnesses"][0]
assert w["probe_hardware"] == "0x1.2c63f40000000p-5"  # FP32 0x3d1631fa
w["sm120_model"] = "0x1.2c63f60000000p-5"            # FP32 0x3d1631fb
c5.validate_results(p, canonical=True)                 # accepted

p = json.loads(c5.CANONICAL_OUTPUT.read_bytes())
p["mma"]["audit_seeds"].clear()
p["derived"] = c5.derive(p)
c5.validate_results(p, canonical=True)                 # accepted with 123,934 dots
```

The validator sums witness pass flags and checks aggregate equality against the remaining denominators. It also accepted the wrong raw NaN word `7ffffffe` with the two pass flags left true, removal of all edge cases, and replacement of one stress category with a duplicate passing row. Requiring the expected population per identity and checking recorded model/hardware words wherever the artifact contains them would close these reproducible gaps.

The broader baseline probes also still accepted an arbitrary 64-hex hardware digest, an arbitrary legacy blob ID, and a recomputed run hash for a false commit/source pair. These require independent evidence, supplied above for the published file. As before, a one-binary64-ULP timing perturbation lay within tolerance, and scaling both members of a nonmedian pair preserved the checked summaries. The clean-tree host guard accepted an untracked file and rejected a tracked edit, an sm_89 capability and CPU execution.

### Timing reconstruction

The ordinary cells contain 4,800 durations (30 × 40 × two variants × two clocks). The Opus cells contain 144 durations (18 × four × two variants). For every cell I counted first and second positions and both within-round predecessor orders. I computed marginal medians and their ratios, then exponentiated the mean log of the within-round ratios. The paired 95% interval used the sample standard deviation of the log ratios and a Student t quantile with 39 or 3 degrees of freedom. The largest absolute difference in 522 numeric checks was **8.53 × 10⁻¹⁴**. A separate reconstruction of the full derived tree from the per-cell observations, raw timing samples and MMA rows matched all 137 leaves. Timings below are recalculations of the recorded v2 run.

| Shape | Kernel | Dtype | Median speedup | Paired geometric mean | Paired 95% interval |
|---|---|---|---:|---:|---:|
| 4×33554432 | PS3_55 | FP16 | 28.3025× | 28.4359× | 28.0165–28.8615× |
| 4×33554432 | PS3_55 | BF16 | 27.9831× | 27.9186× | 26.7044–29.1879× |
| 4×33554432 | SS3_54 | FP16 | 28.5747× | 28.5667× | 28.1277–29.0124× |
| 4×33554432 | SS3_54 | BF16 | 28.3852× | 28.5291× | 27.8960–29.1766× |
| 4×33554432 | SS3_53 | FP16 | 28.3544× | 28.4845× | 28.1296–28.8439× |
| 4×33554432 | SS3_53 | BF16 | 28.4354× | 28.5260× | 28.2216–28.8338× |
| 33554432×2 | PS3_55 | FP16 | 83.4865× | 83.5857× | 82.4108–84.7773× |
| 33554432×2 | PS3_55 | BF16 | 73.6452× | 73.7555× | 73.0712–74.4462× |
| 33554432×2 | SS3_54 | FP16 | 83.9728× | 84.1430× | 83.0595–85.2406× |
| 33554432×2 | SS3_54 | BF16 | 74.0047× | 74.2014× | 73.4796–74.9304× |
| 33554432×2 | SS3_53 | FP16 | 83.5038× | 83.8073× | 82.8532–84.7723× |
| 33554432×2 | SS3_53 | BF16 | 73.9688× | 74.1082× | 73.4869–74.7347× |
| 1×33554432 | PS3_55 | FP16 | 36.5269× | 36.7124× | 36.0685–37.3678× |
| 1×33554432 | PS3_55 | BF16 | 36.2017× | 36.3675× | 35.2868–37.4813× |
| 1×33554432 | SS3_54 | FP16 | 36.6558× | 36.8153× | 36.0302–37.6176× |
| 1×33554432 | SS3_54 | BF16 | 36.6992× | 36.8564× | 36.2882–37.4335× |
| 1×33554432 | SS3_53 | FP16 | 36.4742× | 36.6618× | 35.9393–37.3989× |
| 1×33554432 | SS3_53 | BF16 | 36.7520× | 36.9415× | 36.4081–37.4826× |

### Regression suite

On the RTX 5080 (sm_120, driver 615.78.08), the four requested modules completed with **402 passed, 0 failed, 0 skipped, 104 warnings in 604.67 seconds**. The JUnit records confirm 49 C5-01 tests, 153 SchattenSwitch tests, 169 PolarSwitch Blackwell tests and 31 C4-03 drift tests. The six harvested-trajectory cases all passed in that run, covering FP16 SS3_34, SS3_35, SS3_53 and SS3_54 and BF16 SS3_34 and SS3_35. The fixture read the existing 224,163,445-byte trajectory file through a temporary worktree symlink. The suite used the requested main-project virtual environment with `PYTHONPATH=.`. Independent correctness probes overlapped part of the run.

## Finding 5 Validator Verification Addendum

**FAIL — the hold on canonical promotion remains.** At `f17b26e0db7d0aee6fee9e1f21d25d1cd19b6048`, canonical validation rejected all six previously reported mutations. It still accepted six independently malformed payloads. One changed the FP16 hypothesis witness by one FP32 bit while retaining its claim of equality with the recorded hardware.

The direct parent is `4057daf01b5db502eab8c53e3f458695e9cc94e7`. The only changed files are `research/polar_opt/bench_c5_01_syrk_and_mma.py` and `tests/test_c5_01_syrk_and_mma.py`. The kernels, MMA model and canonical JSON therefore retain their audited Git blobs. The unmodified JSON passed `validate_results(p, canonical=True)` and had SHA-256 `52821ce629c8dca05bd22bab08088521a30a02b6ea23a70fb440bc6ec3b80271`.

The independent CPU run made **22 probes** against fresh parses of that JSON. The validator accepted the one valid control. It rejected **6/6 prior mutations** and **9/15 additional mutations**. The six prior cases covered the one-bit model and SYRK config changes, cleared audit seeds, cleared edge cases, the wrong NaN word `0x7ffffffe` with both flags true, and a duplicated stress row. The nine additional rejections covered an empty exponent table, a changed exponent probe word, empty or duplicated SYRK configs, empty or renamed hypotheses, a missing chain, a changed chain BK, and a non-FP32 witness value distinguishable in binary64.

Each row below gives an independently **accepted** mutation. Start with `p = json.loads(c5.CANONICAL_OUTPUT.read_bytes())` and `m = p["mma"]`. Where indicated, run `p["derived"] = json.loads(json.dumps(c5.derive(p)))` before `c5.validate_results(p, canonical=True)`.

| Accepted mutation | Recompute derived | Contradiction |
|---|---|---|
| `m["hypotheses"][0]["witness"] = "0x1.2c63f60000000p-5"` | No | The SM120 hypothesis reports FP32 word `0x3d1631fb`. The FP16 hardware witness is `0x3d1631fa`, and `witness_equals_hardware` is still true. |
| `m["hypotheses"][0]["bit_matches"] = 2049` | Yes | The hypothesis claims 2,049 matches among 2,048 cases. |
| Set `m["stress"][0]["scalar_crosscheck"] = [511, 512]` and set `scalar_crosscheck = [2, 1]` on the first stress row with `kind == "chain"`. | Yes | The chain's impossible count cancels a real one-count deficit. The canonical scalar promotion check passes. |
| `m["witnesses"][0]["sm120_model"] = "0x1.2c63f40000000001p-5"` | No | The literal exceeds the hardware value by exactly 2⁻⁶⁹. Conversion to binary64 loses that difference before the exact-FP32 check. |
| `m["witnesses"][0]["row"] = [0.0] * 16` | No | The all-zero diagonal still carries the original nonzero model and hardware outputs. |
| `m["exponent_table"][0]["b"] = 0.0` | No | The named one-product case now multiplies by zero, yet still reports 2⁻¹²³. |

With `CUDA_VISIBLE_DEVICES=""` and `PYTHONPATH=.`, the requested `tests/test_c5_01_syrk_and_mma.py -q` run finished with **32 passed, 0 failed, 17 skipped** in 4.31 seconds. All skips required the RTX 5080. This re-audit exercised the validator on CPU.

## Final Finding 5 Validator Verification Addendum

**FAIL — the Finding 5 hold remains.** At `2567991d943fb609251fc167590bdbea8a671ca9`, canonical validation rejected all 12 previously reported mutations. It still accepted two FP16 hypotheses that each claimed 2,048/2,048 hardware matches on the same inputs, even though their predicted FP32 words differ on 239 inputs.

Only `research/polar_opt/bench_c5_01_syrk_and_mma.py` and `tests/test_c5_01_syrk_and_mma.py` changed from `4057daf01b5db502eab8c53e3f458695e9cc94e7`. The kernels, MMA model and canonical JSON retain their audited Git blobs. The unmodified JSON passed canonical validation with SHA-256 `52821ce629c8dca05bd22bab08088521a30a02b6ea23a70fb440bc6ec3b80271`.

| Independent CPU probes | Accepted | Rejected |
|---|---:|---:|
| Unmodified control | 1 | 0 |
| Six mutations at lines 1559–1564 | 0 | 6 |
| Six earlier Finding 5 mutations | 0 | 6 |
| Additional MMA mutations | 4 | 18 |
| **Total** | **5** | **30** |

Each row below reproduces one accepted mutation. Start each with a fresh parse of the canonical JSON as `p` and set `m = p["mma"]`. After the assignment, run `p["derived"] = json.loads(json.dumps(c5.derive(p)))` and `c5.validate_results(p, canonical=True)`.

| Accepted assignment | Evidence |
|---|---|
| `m["hypotheses"][1]["bit_matches"] = 2048` | The FP16 SM120 and C4-03 rows now both claim every hardware match. CPU replay of both model rules on the producer's seeded inputs gave 1,809/2,048 equal predictions. At zero-based index 1, SM120 predicted `0x3a86a61e` and C4-03 predicted `0x3a86a61f`. A single hardware word cannot equal both. |
| `m["syrk_configs"][0]["hardware_sha256"] = "0" * 64` | Configs 0 and 1 for the same FP16 9×2033 input both claim 45/45 bitwise matches against the same ordered model vector. Their hardware vectors must therefore have identical digests. Config 1 retains `6d80f53bce7a8fc863cd0ebcd7ba746b842d60ec71a93e37be3fdf3ee12befa3`. |
| `m["model"]["zero"] = "-0"` | The model now specifies negative zero, while its recorded zero-producing edge cases include positive zero (`0x00000000`). |
| `m["hypotheses"][2]["options"]["guard"] = True` | The one-guard-bit option accepts a JSON boolean in its integer slot. Python compares it equal to 1. |

The 18 additional rejections covered two offsetting fused-store count mutations, six boolean substitutions in counts, a changed model guard width, PTX at all three recording levels, the edge accumulator, both C4-03 model fields, the fused BF16 store, and the hypothesis options and witness equality flag.

With `CUDA_VISIBLE_DEVICES=""` and `PYTHONPATH=.`, the focused pytest module finished with **32 passed, 0 failed, 17 GPU tests skipped in 4.85 seconds**. The hypothesis replay used CPU model predictions on the shared inputs and required no new hardware measurements.

## Finding 5 CPU recheck at `86d7bb0` (v5)

**FAIL — the promotion hold remains.** The unmodified control passed and canonical validation rejected all 16 prior mutations. It accepted 8 of 24 additional mutations. In the clearest case, a finite graph replay claimed bitwise equality and a maximum absolute difference of 1.0 for the same outputs.

| Independent CPU probes reaching `validate_results` | Accepted | Rejected |
|---|---:|---:|
| Unmodified canonical control | 1 | 0 |
| Prior Finding 5 mutations (v2, v3, v4) | 0 | 16 (6 + 6 + 4) |
| Additional MMA | 0 | 7 |
| Additional long-K | 4 | 1 |
| Additional timing | 1 | 2 |
| Additional zero regression | 1 | 2 |
| Additional metadata | 2 | 1 |
| Additional protocol | 0 | 3 |
| **Total** | **9** | **32** |

To reproduce each acceptance below, start with `p = json.loads(c5.CANONICAL_OUTPUT.read_bytes())` and `lk = p["long_k"]`. Apply one edit, set `p["derived"] = json.loads(json.dumps(c5.derive(p)))`, then call `c5.validate_results(p, canonical=True)`. Each call returned normally.

| Accepted edit | Evidence |
|---|---|
| `lk["graphs"][0]["max_abs_diff"] = 1.0` | Both `finite` and `bitwise_equal` remain true. Equal finite outputs have zero difference. |
| `lk["inputs"][0]["finite_per_matrix"] = []` | Shape `[100000, 4]` contains one matrix and requires one finiteness result. |
| `lk["inputs"][0]["sync_free"] = None` | This ordinary shape calls `sync_free` in the producer. The edit removes its measurement and reduces the promotion count from 34/34 to 33/33. |
| `lk["opus_l6"][0]["native"]["sigma_min"] = 2.0` | The minimum singular value now exceeds the recorded maximum of 0.9284811137140414. |
| `p["timing"]["cells"][0]["summary"]["gpu"]["paired"]["n"] = 40.00000001` | The row contains exactly 40 paired rounds. The numeric tolerance accepts a fractional sample count. |
| `p["zero_regression"]["legacy_blobs"]["kernels_blackwell"] = "0" * 40` | Git identifies the blob at the recorded legacy base as `d1ecac7fba2f6b1809e0327b000211c2204f5b51`. This reproduces the earlier provenance gap. |
| `p["meta"]["total_seconds"] = -1.0` | The run reports negative elapsed time despite positive section durations. |
| `del p["meta"]["section_seconds"]["mma"]` | The MMA results remain present, but their section duration is missing from the producer's four-section timing record. |

The seven additional MMA rejections covered whole-group fake digests for SYRK and audit seeds, a negative-zero witness lane, a false vectorized equality flag, an off-by-one BF16 alternative hypothesis count, an extra fused-store count and a missing edge accumulator. The three protocol mutations used recomputed run hashes. A separate deletion of a stress row's `scalar_crosscheck` raised `KeyError` during derivation and is excluded from the validator counts.

The diff from `4057daf01b5db502eab8c53e3f458695e9cc94e7` to `86d7bb08ad7fd3d2c1d125eb4a3d815552b8be03` changes only `research/polar_opt/bench_c5_01_syrk_and_mma.py` and `tests/test_c5_01_syrk_and_mma.py`. All kernels, `blackwell_mma_model.py` and the canonical result retain their audited Git blobs. The result file's SHA-256 matched `52821ce629c8dca05bd22bab08088521a30a02b6ea23a70fb440bc6ec3b80271`. With CUDA devices hidden, the focused pytest module finished with **32 passed, 0 failed, 17 RTX 5080 tests skipped in 13.81 seconds**.

## Finding 5 CPU recheck at `a7b15ad` (v6)

**FAIL — the promotion hold remains.** Canonical validation accepted 6 of 33 fresh malformed payloads. Two independently assign different minimum and maximum singular values to a one-row matrix, which has only one singular value. The unmodified control passed. All 24 prior mutations were rejected.

| Independent CPU probes | Accepted | Rejected |
|---|---:|---:|
| Unmodified canonical JSON | 1 | 0 |
| Prior v2, v3, v4 and v5 mutations | 0 | 24 (6 + 6 + 4 + 8) |
| Fresh metadata | 3 | 5 |
| Fresh protocol, zero regression and timing | 0 | 8 (2 + 3 + 3) |
| Fresh long-K | 3 | 8 |
| Fresh MMA and derived | 0 | 6 (3 + 3) |
| **Total** | **7** | **51** |

To reproduce each acceptance, start with a fresh `p = json.loads(c5.CANONICAL_OUTPUT.read_bytes())` from `research.polar_opt.bench_c5_01_syrk_and_mma` imported as `c5`. Set `lk = p["long_k"]` and `g = p["meta"]["gpu_telemetry"]`. Apply one edit below, set `p["derived"] = json.loads(json.dumps(c5.derive(p)))`, then call `c5.validate_results(p, canonical=True)`. All six calls returned normally.

| Accepted edit | Contradiction |
|---|---|
| `lk["opus_l6"][12]["native"]["sigma_min"] = 0.5` | Shape `[1, 33554432]` has one singular value. Its recorded maximum is 1.016177322736313. |
| `lk["grid"][96]["cells"]["ps3_55_fp16"]["sigma_min"] = 0.5` | Shape `[1, 1048576]` also has one singular value. Its recorded maximum is 1.016177838600595. |
| `lk["inputs"][24]["normal_quality"]["rel_fro_err"] = 10.0` | The normal matrix has four rows and reports `sigma_max = 0.9490285658243612`. Its polar factor has Frobenius norm 2. The triangle inequality bounds the relative error by `1 + sigma_max`, below 1.95. |
| `for side in g.values(): side["time"] += 86400` | Both telemetry samples move one day beyond their recorded run times while their interval stays the same. |
| `g["end"]["utilization.gpu"] = 101.0` | GPU utilization exceeds 100 percent. |
| `del g["start"]["time"]` | The sample retains its measurements, but every successful sample from the recorded producer includes an epoch timestamp. |

Fresh rejections included a fake source SHA-256 with a matching run hash, changed protocol fields with recomputed run hashes, missing and extra keys, corrupted view strides, negative and above-ceiling boundary singular values, mixed-row flag contradictions, NaN and boolean grid metrics, and a floating-point paired sample count. Every source mutation was followed by re-derivation. The three direct derived-field probes edited the freshly recomputed summary immediately before validation. All 51 rejections raised `ValueError` in the validator.

The diff from `4057daf01b5db502eab8c53e3f458695e9cc94e7` to `a7b15ad7aebb98f88611b3a1328b192c9f2aedbd` changes only `research/polar_opt/bench_c5_01_syrk_and_mma.py` and `tests/test_c5_01_syrk_and_mma.py`. The kernels, MMA model and canonical JSON retain their audited Git blobs. The JSON SHA-256 matched `52821ce629c8dca05bd22bab08088521a30a02b6ea23a70fb440bc6ec3b80271`. With CUDA devices hidden, the focused pytest module finished with **32 passed, 0 failed, 17 RTX 5080 tests skipped in 13.71 seconds**.

## Finding 5 CPU recheck at `5a64060` (v7)

**FAIL — keep the promotion hold.** The unmodified canonical JSON passed. The validator rejected all 30 prior mutations and accepted 6 of 41 fresh malformed payloads. One accepted sample lasted 60 seconds inside a 9.973-second section. Two accepted polar errors exceed the mathematical maximum for their recorded singular values.

| Independent CPU probes | Accepted | Rejected |
|---|---:|---:|
| Unmodified canonical control | 1 | 0 |
| Prior v2 through v6 mutations | 0 | 30 (6 + 6 + 4 + 8 + 6) |
| Fresh metadata | 3 | 13 |
| Fresh protocol and zero regression | 0 | 5 (2 + 3) |
| Fresh timing | 1 | 2 |
| Fresh long-K | 2 | 9 |
| Fresh MMA and derived summaries | 0 | 6 (3 + 3) |
| **Total** | **7** | **65** |

To reproduce any acceptance, import `json` and `from research.polar_opt import bench_c5_01_syrk_and_mma as c5`. Start each row with a fresh `p = json.loads(c5.CANONICAL_OUTPUT.read_bytes())`. Set `meta = p["meta"]`, `ss = meta["section_seconds"]`, `t = p["timing"]["cells"][0]` and `lk = p["long_k"]`. Apply only the edit in that row.

| Accepted edit | Evidence |
|---|---|
| `meta["command"].append("--help")` | The recorded driver's argument parser prints help and exits before the benchmark can run. |
| `meta["environment"]["env"]["PYTHONPATH"] = "." + chr(0) + "x"` | A Linux process environment value cannot contain an embedded NUL. |
| `ss["mma"] += ss["timing"] - 1.0`, then `ss["timing"] = 1.0` | The sum of section durations is preserved. The timing section now claims one second for sequential wall-clock samples that alone total 7.618142 seconds. |
| `t["rounds"][0]["native"]["wall_ms"] = 60000.0`, then `t["summary"] = c5.timing_summary(t["rounds"])` | A single sample exceeds the recorded 9.972741-second timing section. Both durations come from the producer's performance counter. |
| `lk["opus_l6"][6]["control"].update(sigma_min=0.25, sigma_max=2.0, rel_fro_err=2.35)` | Shape `[33554432, 2]` fixes both singular values. Their relative polar error cannot exceed 2.298098. |
| `lk["inputs"][24]["normal_quality"].update(sigma_min=0.25, sigma_max=1.0, rel_fro_err=1.85)` | The normal matrix has four rows. Even with both middle singular values at 1, the relative polar error cannot exceed 1.841365. |

For the metadata probes, refresh `p["run_config_hash"] = c5.run_config_hash(p["quick"], p["protocol"], meta["code_sha256"], meta["git_sha"], meta["environment"])`. For every row, finish with `p["derived"] = json.loads(json.dumps(c5.derive(p)))` and `c5.validate_results(p, canonical=True)`. All six calls returned normally.

The two polar bounds follow from the singular values of the polar factor, which are all 1. Von Neumann's trace inequality gives `||Q - P||_F² ≤ Σᵢ(σᵢ(Q) + 1)²`. For the producer's relative error, divide by the short dimension and take the square root. Given only the extreme singular values, the largest upper bound places every remaining value at the maximum.

All 65 rejections raised `ValueError` in the validator after successful derivation. Fresh rejections included rank-one and rank-two singular-value contradictions, an Opus/grid overlap mismatch, mixed-input finiteness and zero-flag conflicts, and a GPU duration above its wall duration after timing-summary repair. Telemetry rejections covered epochs, both clocks, temperature, power, memory, P-state, throttle mask and error keys. Protocol edits used recomputed run hashes. Direct derived-field edits followed fresh derivation.

The Git tree diff from `4057daf01b5db502eab8c53e3f458695e9cc94e7` to `5a64060fb464ba8659918d8b4708ca798224a651` contains only the benchmark driver and its focused test module. The kernels, MMA model and published JSON retain their audited blobs. The JSON SHA-256 matched `52821ce629c8dca05bd22bab08088521a30a02b6ea23a70fb440bc6ec3b80271`. With CUDA devices hidden, the focused pytest module finished with **32 passed, 0 failed, 17 RTX 5080 tests skipped in 14.01 seconds**. The hardware evidence remains the live audit at `fa8f24dd1bae83669abd9811af1a3b37c0fef496`.

## Finding 5 CPU recheck at `f93e9a1` (v8)

**FAIL — keep the promotion hold.** The unmodified canonical artifact passed. All 36 prior mutations raised `ValueError`. The validator accepted four new contradictions in the SYRK accuracy result, Opus speedup, source provenance and cache-flush protocol.

| Independent CPU probes | Accepted | Rejected |
|---|---:|---:|
| Unmodified canonical JSON | 1 | 0 |
| Prior Finding 5 mutations, v2–v7 | 0 | 36 (6 + 6 + 4 + 8 + 6 + 6) |
| Fresh mutations | 4 | 11 |
| **Total** | **5** | **47** |

To reproduce, import `json` and `from research.polar_opt import bench_c5_01_syrk_and_mma as c5`. For each row, start with `p = json.loads(c5.CANONICAL_OUTPUT.read_bytes())` and `r = p["long_k"]["opus_l6"][0]`. Apply just that row, then run the common validation code below. All four calls returned normally.

| Accepted edit | Evidence |
|---|---|
| `p["long_k"]["syrk"][0]["diag_bias_max"] = 1e20` | The reported worst diagonal bias rises from 0.003596 to 10²⁰. In this FP16-input, FP16-output row, the producer measures `abs(1 - out / exact)`. A finite FP16 output has magnitude at most 65,504. A positive Gram diagonal sums squares and is at least 2⁻⁴⁸, the square of the smallest positive FP16 value. A zero denominator cannot yield a finite bias. Even the loose bound 1 + 65,504 × 2⁴⁸ is below 10²⁰. |
| Set `s["native"]["wall_ms"] = 1e-12` for every `s` in `r["samples"]`. Set `r["speedup"] = c5.speedup(r["samples"])` and `r["speedup_paired"] = c5.paired_speedup(r["samples"])`. | The four synchronized native calls each claim one femtosecond on the recorded Linux platform. The producer uses CPython’s monotonic performance counter. The CPU check reports one-nanosecond clock resolution. The accepted median speedup rises from 28.30× to 1.73 × 10¹⁴×. |
| `p["meta"]["git_sha"] = "7d960703b674a14d6c173855a76d3e5f9b0ca64a"` | `git cat-file -t` identifies this object as the root tree of the real producer commit `e750287`. The source paths resolve and their digest matches, but a tree has no commit identity or history. The producer records `git rev-parse HEAD`. The validator accepts the tree with run hash `4f8f75b50f138d87`. |
| `p["timing"].update(l2_cache_bytes=1, flush_bytes=4)` | The producer sizes the flush from the device property. The audited RTX 5080 result records 67,108,864 bytes of L2 and a 268,435,456-byte flush. Preserving the 4× ratio lets a four-byte write qualify as a full cache flush for every ordinary timing cell. |

```python
m = p["meta"]
p["run_config_hash"] = c5.run_config_hash(
    p["quick"], p["protocol"], m["code_sha256"], m["git_sha"], m["environment"]
)
p["derived"] = json.loads(json.dumps(c5.derive(p)))
c5.validate_results(p, canonical=True)
```

All 11 fresh rejections raised `ValueError` after derivation. They covered missing or duplicated population rows in four sections, forged stress and source digests, a nonfinite graph, impossible polar error, unequal launch counts, negative diagonal bias and Opus samples longer than the long-K section.

The diff from `4057daf01b5db502eab8c53e3f458695e9cc94e7` to `f93e9a11c8b3933e42c4e107a82f96a92398fcca` changes only the benchmark driver and its focused test module. All kernels, the MMA model and the JSON retain their audited Git blobs. The artifact SHA-256 matched `52821ce629c8dca05bd22bab08088521a30a02b6ea23a70fb440bc6ec3b80271`. With CUDA devices hidden, the focused module finished with **32 passed, 0 failed, 17 RTX 5080 tests skipped in 14.26 seconds**. The hardware evidence comes from the live audit at `fa8f24dd1bae83669abd9811af1a3b37c0fef496`.

## Finding 5 final CPU verification at `18aba75` (v9)

**FAIL — keep the promotion hold.** The committed artifact passed unchanged. All 40 prior mutations and all seven sibling probes raised `ValueError`. Four fresh, independently accepted payloads contradicted the producer or the recorded hardware.

| Independent CPU probes | Accepted | Rejected |
|---|---:|---:|
| Unmodified canonical artifact | 1 | 0 |
| Prior Finding 5 mutations, v2–v8 | 0 | 40 (6 + 6 + 4 + 8 + 6 + 6 + 4) |
| BF16 bias ceiling, separate GPU and wall clock floors, hidden CUDA, RTX 5080 L2 size, short-K launch mismatch and long-K launch undercount | 0 | 7 |
| Fresh scientific, population and provenance checks | 4 | 8 |
| Informational P-state spelling check | 1 | 0 |
| Total | 6 | 55 |

To reproduce each blocker, start with a fresh parse of the canonical artifact and apply exactly one edit from the table. Then run the three verification statements below. All four calls returned normally.

| Accepted edit | Contradiction |
|---|---|
| `p["long_k"]["opus_l6"][12]["control"].update(sigma_min=1e100, sigma_max=1e100, rel_fro_err=1e100)` | The derived maximum control singular value rises from 1.186664 to 10¹⁰⁰. This PS3_55 call returns a finite FP32 row of length 33,554,432. Its only singular value is its Euclidean norm, at most √33,554,432 × FP32_MAX < 1.98 × 10⁴². |
| `p["long_k"]["launches_at_graph_shape"]["native"]["ps3_55"] = 17` | The recorded count is 26. At shape [4, 131072], each of two PS3_55 phases launches five symmetric products, a matrix multiply, and at least three chunked Gram kernels (partial SYRK, reduction and FP16 store). Those operations alone require at least 18 launches. |
| `p["long_k"]["inputs"][24]["finite_per_matrix"][3:] = [True, True]` | This FP16 PS3_55 row now reports finite outputs for the NaN and infinity inputs. With NaN the Frobenius norm is NaN. With infinity the normalization includes Inf/Inf. PS3_55 propagates those non-finite values through its Gram and matrix products. The live audit recorded both outputs as non-finite. |
| `p["meta"]["environment"]["capability"] = [13, 0]` | With the refreshed run hash, the validator attributes an sm_130 run to the named NVIDIA GeForce RTX 5080. That device has compute capability 12.0, as recorded in the audited artifact. |

```python
# Start each probe with p = json.loads(c5.CANONICAL_OUTPUT.read_bytes()).
# Apply one edit from the table, then run these three statements.
p["run_config_hash"] = c5.run_config_hash(p["quick"], p["protocol"], p["meta"]["code_sha256"],
                                          p["meta"]["git_sha"], p["meta"]["environment"])
p["derived"] = json.loads(json.dumps(c5.derive(p)))
c5.validate_results(p, canonical=True)
```

The eight fresh rejections covered four omitted population rows (timing, audit seed, grid and boundary), forged stress and source digests, conflicting Opus/grid quality, and negative SYRK bias. Every rejection occurred in the validator after successful derivation. As a non-blocking metadata observation, a zero-padded P-state spelling passed with the same numeric P-state.

The diff from `4057daf01b5db502eab8c53e3f458695e9cc94e7` to `18aba75bc31723c019341514816cae0a3c19d9b1` contains only the benchmark driver and its focused test module. The kernels, MMA model and results JSON retain their audited blobs. The JSON SHA-256 matched `52821ce629c8dca05bd22bab08088521a30a02b6ea23a70fb440bc6ec3b80271`. With CUDA devices hidden, pytest reported **32 passed, 0 failed, 17 GPU-only skips in 14.26 seconds**. The RTX 5080 measurements come from the live audit at `fa8f24dd1bae83669abd9811af1a3b37c0fef496`.

## Finding 5 closed promotion signoff at `9358270` (v10)

**PASS — HOLD LIFTED.** The unchanged canonical artifact passed `validate_results(p, canonical=True)`. All 52 negative probes raised `ValueError` inside that validator after successful re-derivation. Each probe started from a fresh parse, refreshed `run_config_hash`, and recomputed `derived` through a JSON round trip.

| Closed CPU replay | Rejected / tested |
|---|---:|
| Previously reported mutations, v2–v8 | 40/40 |
| Four v9 blockers at lines 1741–1744 | 4/4 |
| Informational v9 P-state spelling `P08` | 1/1 |
| Seven v9 sibling probes | 7/7 |

Only the benchmark driver and its focused test module changed from `4057daf` to `9358270`. The kernels, MMA model, and result JSON retain their live-audited Git blobs. The result SHA-256 matched `52821ce629c8dca05bd22bab08088521a30a02b6ea23a70fb440bc6ec3b80271`. With CUDA hidden, the focused pytest module finished with **32 passed, 0 failed, 17 GPU-only skips**.

The `validate_results` docstring defines the offline checks on producer JSON structure, Git commit provenance, mathematical consistency, and promotion gates. Independent hardware evidence for the kernel, MMA, rounding, timing, and digest claims comes from the RTX 5080 replay recorded at `fa8f24dd1bae83669abd9811af1a3b37c0fef496`. Together these checks close Finding 5 for C5-01.
