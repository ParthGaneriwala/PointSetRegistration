Search.setIndex({"alltitles": {"(https://siavashk.github.io/2017/05/14/coherent-point-drift/)": [[1, "https-siavashk-github-io-2017-05-14-coherent-point-drift"]], "Algorithm Overview": [[1, "algorithm-overview"]], "Background": [[1, "background"]], "Coherent Point Drift (CPD) Algorithm": [[1, "coherent-point-drift-cpd-algorithm"]], "Comparing Coherent Point Drift (CPD) and Iterative Closest Point (ICP) algorithms provides insights into their respective strengths and weaknesses in point cloud registration tasks. CPD, a probabilistic framework, excels in scenarios where point correspondences are ambiguous or partial, thanks to its ability to model soft correspondences and account for noise and outliers effectively. It can handle non-rigid transformations and deformations, making it suitable for tasks involving complex shapes or objects undergoing morphological changes. However, CPD may suffer from increased computational complexity, especially with large datasets, and requires careful parameter tuning for optimal performance. On the other hand, ICP is a simpler and computationally efficient method, ideal for rigid transformations and scenarios with well-defined point correspondences. It converges quickly and is robust to noise, making it suitable for real-time applications or situations where computational resources are limited. Nevertheless, ICP\u2019s rigid assumption limits its applicability to non-rigid transformations or scenarios with significant shape variations. In summary, while CPD offers versatility and robustness for challenging registration tasks, ICP provides simplicity and efficiency for more straightforward alignment problems. The choice between CPD and ICP depends on the specific requirements of the registration task, including the nature of the point clouds, the presence of noise or outliers, and the desired level of computational complexity.": [[1, "comparing-coherent-point-drift-cpd-and-iterative-closest-point-icp-algorithms-provides-insights-into-their-respective-strengths-and-weaknesses-in-point-cloud-registration-tasks-cpd-a-probabilistic-framework-excels-in-scenarios-where-point-correspondences-are-ambiguous-or-partial-thanks-to-its-ability-to-model-soft-correspondences-and-account-for-noise-and-outliers-effectively-it-can-handle-non-rigid-transformations-and-deformations-making-it-suitable-for-tasks-involving-complex-shapes-or-objects-undergoing-morphological-changes-however-cpd-may-suffer-from-increased-computational-complexity-especially-with-large-datasets-and-requires-careful-parameter-tuning-for-optimal-performance-on-the-other-hand-icp-is-a-simpler-and-computationally-efficient-method-ideal-for-rigid-transformations-and-scenarios-with-well-defined-point-correspondences-it-converges-quickly-and-is-robust-to-noise-making-it-suitable-for-real-time-applications-or-situations-where-computational-resources-are-limited-nevertheless-icp-s-rigid-assumption-limits-its-applicability-to-non-rigid-transformations-or-scenarios-with-significant-shape-variations-in-summary-while-cpd-offers-versatility-and-robustness-for-challenging-registration-tasks-icp-provides-simplicity-and-efficiency-for-more-straightforward-alignment-problems-the-choice-between-cpd-and-icp-depends-on-the-specific-requirements-of-the-registration-task-including-the-nature-of-the-point-clouds-the-presence-of-noise-or-outliers-and-the-desired-level-of-computational-complexity"]], "Expectation Step": [[1, "expectation-step"], [1, "id1"]], "GMM-based Registration": [[1, "gmm-based-registration"]], "Gaussian Mixture Models": [[1, "gaussian-mixture-models"]], "ICP Algorithm": [[1, "icp-algorithm"]], "Iterative Closest Point (ICP) Algorithm": [[1, "iterative-closest-point-icp-algorithm"]], "Maximization Step": [[1, "maximization-step"], [1, "id2"]], "Missing Correspondences": [[1, "missing-correspondences"]], "Parth Ganeriwala Comprehensive Breadth Examination": [[0, "parth-ganeriwala-comprehensive-breadth-examination"]], "Point Cloud Registration with CPD Example Code": [[1, "point-cloud-registration-with-cpd-example-code"]], "Point Set Registration": [[1, "point-set-registration"]], "Tutorial: Coherent Point Drift (CPD) Algorithm for Point Set Registration": [[1, "tutorial-coherent-point-drift-cpd-algorithm-for-point-set-registration"]]}, "docnames": ["intro", "tutorial"], "envversion": {"sphinx": 61, "sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1}, "filenames": ["intro.md", "tutorial.ipynb"], "indexentries": {}, "objects": {}, "objnames": {}, "objtypes": {}, "terms": {"0": 1, "000001": 1, "0005693709355462084": 1, "001": 1, "00117814948": 1, "0013629466508553016": 1, "002393868947999754": 1, "00252268": 1, "00288": 1, "00342736": 1, "0037422437578762038": 1, "005": 1, "005641678099": 1, "00577778359249276": 1, "008740422079077206": 1, "01": 1, "011183287215": 1, "0126": 1, "01278116": 1, "012944361024599126": 1, "016735989286075965": 1, "01908014768237006": 1, "023039533118": 1, "024407959105": 1, "028501096352114078": 1, "031133102868": 1, "04319878670547142": 1, "04d": 1, "05": 0, "0505037209060204": 1, "0581179899507": 1, "062884468844": 1, "06510246805138603": 1, "065410320334": 1, "0655096113323": 1, "078324329664": 1, "080687619164": 1, "083111354254": 1, "086929721825": 1, "0905": 0, "09491426465540308": 1, "1": 1, "10": 1, "100": 1, "10095": 1, "10603046353128": 1, "10716": 1, "1087657965903": 1, "11": 1, "110": 1, "111": 1, "11173": 1, "112": 1, "113": 1, "114": 1, "11504": 1, "11743": 1, "118637008187": 1, "11913": 1, "12": 1, "120": 1, "12035": 1, "121": 1, "12121": 1, "12182": 1, "122": 1, "12226": 1, "12256": 1, "12277": 1, "12293": 1, "12303": 1, "12310": 1, "12316": 1, "12319": 1, "12322": 1, "12323": 1, "12324": 1, "12325": 1, "12326": 1, "1248391040758": 1, "13": 1, "13024978905": 1, "13166046839863524": 1, "14": 0, "1457": 1, "147202665652": 1, "149": 1, "15": 1, "1520029563609": 1, "15772759593": 1, "16": 1, "16923728827": 1, "17": 1, "17195106812835": 1, "17264469262804974": 1, "175826047194": 1, "1767": 1, "178": 1, "17941056": 1, "18": 1, "188261015732": 1, "19": 1, "19510814338355": 1, "19661950236275": 1, "2": 1, "20": 1, "2009": 1, "2010": 1, "2017": 0, "204": 1, "21": 1, "21443737679234914": 1, "215055268316": 1, "2153717671274": 1, "22": 1, "228": 1, "23": 1, "2310": 1, "2311": 1, "2313": 1, "2314": 1, "23554891626634": 1, "2360673666164": 1, "23628983803764": 1, "238687585734": 1, "239352565915": 1, "24": 1, "24322132156686": 1, "248": 1, "25": 1, "25192770762908": 1, "253263433289": 1, "25437817984755": 1, "2549653927841887": 1, "25526704296226": 1, "26": 1, "262908418941": 1, "2635": 0, "266": 1, "27": 1, "27215216538843": 1, "2791555248095": 1, "279393063454": 1, "279835998941": 1, "28": 1, "280185306256": 1, "281": 1, "29": 1, "291073983992": 1, "293": 1, "29576872984077596": 1, "297": 1, "2d": 1, "3": 1, "30": 1, "305": 1, "3058": 1, "31": 1, "315": 1, "32": 1, "323": 1, "32471561664": 1, "32831023428946": 1, "32857442537977": 1, "33": 1, "330": 1, "335": 1, "3351": 1, "339": 1, "34": 1, "340169890465": 1, "340472645928003": 1, "34193972709176": 1, "342": 1, "3436177679382573": 1, "344": 1, "347": 1, "35": 1, "350": 1, "355": 1, "357836018864": 1, "36": 1, "360": 1, "364": 1, "3651398178228": 1, "36576953891995": 1, "368": 1, "37": 1, "372": 1, "376": 1, "379": 1, "38": 1, "381": 1, "383": 1, "385": 1, "387": 1, "389": 1, "39": 1, "390": 1, "391": 1, "392": 1, "393": 1, "394": 1, "3964319901024": 1, "3d": 1, "3x3": 1, "4": 1, "40": 1, "41": 1, "410672599864": 1, "42": 1, "421975444241549": 1, "4229687577621": 1, "42386992690886": 1, "43": 1, "435640172051": 1, "4393296148282": 1, "44": 1, "44475901819": 1, "45": 1, "454": 1, "455": 1, "456": 1, "46": 1, "47": 1, "48": 1, "49": 1, "49385768308": 1, "5": 1, "50": 1, "500": 1, "50501711223853": 1, "51391244": 1, "515192188104": 1, "5189": 1, "51961453478395": 1, "53": 1, "533180114131": 1, "536568905538": 1, "54275994131405": 1, "54501424138": 1, "55601275532354": 1, "55759368675953": 1, "579322164629": 1, "582707112002": 1, "5853532172715": 1, "586058655266": 1, "5e": 1, "6": 1, "6037314209917": 1, "61872075598933": 1, "619648426247": 1, "6216624753241": 1, "62188664": 1, "6241337948345": 1, "630505704383": 1, "63217528497373": 1, "6382966351424": 1, "638998760804": 1, "639663051990155": 1, "648543556737": 1, "6534": 1, "658602300051": 1, "6671": 1, "671922843918": 1, "6740": 1, "6795": 1, "6823": 1, "684011243648": 1, "6855": 1, "687555303884": 1, "6918": 1, "69427652535": 1, "6983": 1, "7": 1, "7047": 1, "7109": 1, "7166": 1, "718323405797": 1, "72": 1, "7219": 1, "7266": 1, "7277604301511": 1, "7307": 1, "7332542687565": 1, "7343": 1, "7374": 1, "7401": 1, "7423": 1, "7441": 1, "7456": 1, "7469": 1, "7479": 1, "7487": 1, "7494": 1, "7499": 1, "75": 1, "7503": 1, "7507": 1, "7509": 1, "7512": 1, "7513": 1, "7514": 1, "7515": 1, "7516": 1, "7517": 1, "7518": 1, "757184575563": 1, "7679461983787543": 1, "780899412417": 1, "8": 1, "803980803936": 1, "811578291456": 1, "8185": 1, "82": 1, "822878701505": 1, "827055821826": 1, "827716082884": 1, "838901076053": 1, "84": 1, "85": 1, "851372792849": 1, "853034369753": 1, "85933692601992": 1, "86": 1, "87": 1, "8755044536374": 1, "88": 1, "884203132246": 1, "8874767982525": 1, "88925526041015": 1, "894432973448": 1, "894851775329": 1, "9": 1, "90011871599792": 1, "9026686560628": 1, "904239543952": 1, "908171482724": 1, "908837626514": 1, "9186752349679": 1, "92075247235": 1, "921714199025": 1, "9266": 1, "9277572989842": 1, "931296756868": 1, "94": 1, "946763508957": 1, "948802275268": 1, "9522256942594538": 1, "9571653449275": 1, "973459555933": 1, "975335427005": 1, "985355239": 1, "988682907468": 1, "9983546875213": 1, "A": 1, "But": 0, "For": 1, "If": 1, "In": 0, "The": 0, "These": 1, "To": 1, "_": 1, "_callback": 1, "_i": 1, "_j": 1, "_sourc": 1, "_wrapreduct": 1, "aa": 1, "ab": 1, "accord": 1, "accur": 1, "achiev": 1, "across": 1, "actual": 1, "ad": [0, 1], "add": [0, 1], "add_geometri": 1, "add_subplot": 1, "address": 1, "advantag": 1, "affin": 1, "after": 1, "aim": 1, "algorihtm": 0, "algorithm": 0, "all": 1, "allclos": 1, "allow": 1, "alreadi": 1, "also": [0, 1], "altern": 1, "an": 1, "analysi": 1, "andrii": 1, "angl": 1, "ani": 0, "anoth": 1, "appdata": 1, "appli": 1, "approach": 1, "arbitrari": 1, "argmin": 1, "around": 1, "arrai": 1, "articul": 1, "arxiv": 0, "asarrai": 1, "assert": 1, "assign": 1, "assum": 1, "atol": 1, "author": 0, "averag": 1, "ax1": 1, "ax2": 1, "axes3d": 1, "axi": 1, "b": 1, "backend": 1, "background": 0, "bay": 1, "bb": 1, "befor": 1, "being": 1, "below": 1, "best": 1, "best_fit_transform": 1, "beyond": 1, "biolog": 1, "black": 1, "block": 1, "blue": 1, "break": 1, "briefli": 1, "build": 1, "bunni": 1, "c": 1, "calcul": 1, "calculate_correspondence_prob": 1, "call": 1, "callback": 1, "can": 0, "capture_screen_imag": 1, "case": 1, "cb": 1, "cdist": 1, "cell": 1, "center": 1, "centroid": 1, "centroid_a": 1, "centroid_b": 1, "check": 1, "circ": 1, "circl": 1, "close": 1, "closer": 1, "cloud_0": 1, "co": 1, "code": 0, "coeffici": 1, "coher": 0, "coherentpointdrift": 1, "collect": 1, "com": 0, "commonli": 1, "comparison": [0, 1], "compat": 1, "compon": 1, "concept": 0, "concret": 0, "condit": 1, "confid": 1, "consecut": 1, "constant": 1, "constitut": 1, "construct": 1, "continu": 1, "contribut": 1, "copi": 1, "core": 1, "could": 1, "covari": 1, "cpd": 0, "creat": 1, "create_window": 1, "criteria": 1, "cross": 1, "crucial": 1, "current": 1, "cv": 1, "d": 1, "dash": 1, "data": 1, "deal": 1, "debug": 1, "decomposit": 1, "deepcopi": 1, "def": 1, "deg2rad": 1, "den": 1, "denomin": 1, "densiti": 1, "descript": 0, "destin": 1, "det": 1, "detail": 1, "detect": 1, "deviat": 1, "diag": 1, "differ": 1, "dim": 1, "dimens": 1, "dimension": 1, "directli": 1, "disabl": 1, "discuss": 1, "disrupt": 1, "distanc": 1, "distance_modul": 1, "distinct": 1, "distribut": 1, "dive": 1, "divid": 1, "domain": 1, "done": 1, "dot": 1, "drawn": 1, "drift": 0, "drop": 1, "dst": 1, "dst_demean": 1, "dtype": 1, "e": [0, 1], "each": 1, "els": 1, "em": 1, "enabl": 1, "ensur": 1, "environ": 1, "ep": 1, "equat": 0, "error": 1, "establish": 1, "estep_r": 1, "estepresult": 1, "estim": 1, "estimate_norm": 1, "euclidean": 1, "euler": 1, "euler2mat": 1, "even": 1, "exam": 0, "exampl": 0, "exit": 1, "exp": 1, "expect": 0, "expectation_step": 1, "experi": [0, 1], "explain": 0, "explan": 1, "explicitli": 1, "explor": 1, "extend": 1, "factor": 1, "fals": 1, "fig": 1, "figsiz": 1, "figur": 1, "file": 1, "final": 1, "find": [0, 1], "finfo": 1, "first": 1, "fish_sourc": 1, "fish_target": 1, "fit": 1, "fix": 1, "float": 1, "float32": 1, "follow": [0, 1], "format": 1, "frac": 1, "fromnumer": 1, "full_matric": 1, "function": 1, "fundament": 1, "furthermor": 1, "g": 0, "gener": 1, "geometr": 1, "geometri": 1, "get": 1, "getlogg": 1, "github": 0, "given": 1, "go": 1, "goal": 1, "googl": 0, "graphic": 0, "guess": 1, "gui": 1, "h": 1, "ha": 1, "handshak": 1, "have": 1, "here": 0, "homogen": 1, "http": 0, "i": 0, "icp": 0, "icp_iter": 1, "icpconvergencecriteria": 1, "ident": 1, "ij": 1, "imag": 1, "image_": 1, "implement": 1, "implicitli": 1, "import": 1, "impos": 1, "includ": 0, "incorpor": 1, "indic": 1, "infer": 1, "info": 1, "inform": 1, "init_pos": 1, "initi": 1, "input": 1, "instanc": 1, "instead": 1, "introduc": 1, "invers": 1, "io": 0, "isocontour": 1, "isotrop": 1, "j": 1, "jpg": 1, "jupyt": [0, 1], "k": 1, "kdtreesearchparamhybrid": 1, "keepdim": 1, "keyboardinterrupt": 1, "kneighbor": 1, "know": 1, "known": 1, "kwarg": 1, "label": 1, "last": 1, "latex": 0, "least": 1, "legend": 1, "let": 1, "lib": 1, "like": 1, "likelihood": 1, "linalg": 1, "line": 1, "link": 1, "loadtxt": 1, "local": 1, "log": 1, "look": 1, "loop": 1, "loss": 1, "m": 1, "mai": 0, "maintain": 1, "make": 0, "map": 1, "mat2eul": 1, "match": 1, "materi": 0, "mathcal": 1, "mathrm": 1, "matplotlib": 1, "matric": 1, "matrix": 1, "max": 1, "max_iter": 1, "max_nn": 1, "maximis": 1, "maximization_step": 1, "maximum": 1, "maxit": 1, "md": 1, "mean": 1, "mean_error": 1, "medic": 1, "met": 1, "method": 0, "min": 1, "minim": 1, "minima": 1, "minor": 1, "mix": 1, "most": 1, "motion": 1, "move": 1, "mpl_toolkit": 1, "mplot3d": 1, "mstep": 1, "mu_dst": 1, "mu_j": 1, "mu_k": 1, "mu_src": 1, "mui": 1, "multipl": 1, "multipli": 1, "multivair": 1, "must": 1, "mux": 1, "mx1": 1, "mxm": 1, "myronenko": [0, 1], "n": 1, "n_neighbor": 1, "n_p": 1, "n_random": 1, "namedtupl": 1, "ndarrai": 1, "ndim": 1, "nearest": 1, "nearest_neighbor": 1, "nearestneighbor": 1, "need": 1, "neg": 1, "neigh": 1, "neighbor": 1, "newaxi": 1, "noise_amp": 1, "noise_sigma": 1, "none": 1, "nonrigid": 1, "normal": 1, "note": 1, "notebook": 0, "now": 1, "np": 1, "num_test": 1, "number": 1, "numpi": 1, "nxm": 1, "o3": 1, "obj": 1, "observ": 1, "off": 1, "onc": 1, "one": 1, "ones": 1, "onli": 1, "onlin": 0, "onto": 1, "open3d": 1, "open3dvisualizercallback": 1, "order": 1, "org": 0, "orient": 1, "orient_normals_to_align_with_direct": 1, "origin": 1, "orthogon": 1, "otherwis": 1, "our": 1, "out": 1, "output": 1, "over": 1, "overlap": 1, "p": 1, "p1": 1, "packag": 1, "paint_uniform_color": 1, "pair": 1, "paper": 1, "param": 1, "parameter": 1, "passkwarg": 1, "pcd": 1, "pdf": 0, "pi": 1, "pi_j": 1, "pi_k": 1, "pipelin": 1, "plot": [0, 1], "plot2dcallback": 1, "plt": 1, "pmat": 1, "point": 0, "pointcloud": 1, "pointsetregistr": 1, "poll_ev": 1, "pose": 1, "posit": 1, "possibl": [0, 1], "posterior": 1, "power": 1, "predefin": 1, "prepar": 0, "prepare_source_and_target_nonrigid_2d": 1, "prepare_source_and_target_nonrigid_3d": 1, "prepare_source_and_target_rigid_3d": 1, "preserv": 1, "prev_error": 1, "print": 1, "probabl": 1, "probreg": 1, "process": 1, "procrust": 1, "program": 1, "project": [0, 1], "promin": 1, "proxim": 1, "pt1": 1, "px": 1, "py": 1, "pycpd": 1, "pyplot": 1, "python": 1, "python310": 1, "quad": 1, "quaternion": 1, "quit": 1, "r": 1, "r1": 1, "r_": 1, "rad2deg": 1, "radian": 1, "radiu": 1, "rais": 1, "rand": 1, "randn": 1, "random": 1, "rang": 1, "ravel": 1, "re": 1, "reach": 1, "read_point_cloud": 1, "recent": 1, "recognit": 1, "reconstruct": 1, "red": 1, "reduc": 1, "reduct": 1, "refer": 1, "refin": 1, "reflect": 1, "reg_p2p": 1, "registr": 0, "registration_cpd": 1, "registration_icp": 1, "repeat": 1, "repres": 1, "represent": 1, "research": 0, "resourc": 0, "result": 1, "return": 1, "return_dist": 1, "rg": 1, "robot": 1, "rot": 1, "rotat": 1, "rotation_matrix": 1, "row": 1, "run": 1, "ry": 1, "sampl": 1, "save": 1, "save_imag": 1, "scale": 1, "scatter": 1, "search_param": 1, "seen": 1, "self": 1, "sensit": 1, "server": 1, "set": 0, "set_callback": 1, "set_titl": 1, "set_xlabel": 1, "set_ylabel": 1, "set_zlabel": 1, "setlevel": 1, "should": [0, 1], "show": 1, "shown": 1, "shuffl": 1, "siavashk": 0, "side": 1, "sigma": 1, "sigma2": 1, "sigma_j": 1, "sigma_k": 1, "similar": 1, "simpl": 1, "simplifi": 1, "simultan": 1, "sin": 1, "sinc": 1, "singular": 1, "site": [0, 1], "sklearn": 1, "small": 1, "snippet": 1, "so": 1, "solut": 1, "solv": 1, "some": 1, "song": 1, "sourc": 1, "source_filenam": 1, "spatial": 1, "special": 1, "sqeuclidean": 1, "sqrt": 1, "squar": 1, "src": 1, "src_demean": 1, "stack": 1, "standard": [0, 1], "start": [0, 1], "step": 0, "stop": 1, "structur": 1, "sum": 1, "sum_": 1, "sum_p": 1, "summar": 1, "svd": 1, "t": 1, "t1": 1, "t3d": 1, "t_sourc": 1, "take": 1, "taken": 1, "target": 1, "target_filenam": 1, "term": 1, "test": 1, "test_best_fit": 1, "test_cpd": 1, "test_icp": 1, "text": 0, "tf_param": 1, "tf_type_nam": 1, "th": 1, "than": 1, "them": 1, "theorem": 1, "theta": 1, "thi": [0, 1], "though": 1, "three": 1, "threshold": 1, "through": 1, "tile": 1, "tissu": 1, "title_a": 1, "title_b": 1, "togeth": 1, "toi": 1, "tol": 1, "toler": 1, "tool": 1, "topolog": 1, "total_tim": 1, "tp": 1, "tr": 1, "trace": 1, "traceback": 1, "tradit": 1, "transformationestimationpointtopoint": 1, "transforms3d": 1, "translat": 1, "transpos": 1, "trar": 1, "treat": 1, "true": 1, "tutori": 0, "tweak": 1, "two": 1, "txt": 1, "type": 1, "u": 1, "ufunc": 1, "unit": 1, "unknown": 1, "until": 1, "up": 1, "updat": 1, "update_geometri": 1, "update_render": 1, "update_transform": 1, "updatetransform": 1, "updatevari": 1, "us": [0, 1], "use_cuda": 1, "util": 1, "v": 1, "valu": 1, "valueerror": 1, "varianc": 1, "variou": 1, "vector": 1, "vector3dvector": 1, "vert": 1, "vi": 1, "vision": 1, "visual": 1, "visualize_point_cloud": 1, "voxel_down_sampl": 1, "voxel_s": 1, "vstack": 1, "vt": 1, "w": 1, "w_": 1, "wa": 1, "we": 1, "webrtc": 1, "webrtcwindowsystem": 1, "websit": 0, "webvisu": 1, "weigh": 1, "weight": 1, "were": 1, "what": 1, "when": 1, "whenev": 0, "whether": 1, "which": 1, "wide": 1, "word": 1, "work": 1, "would": 1, "written": 0, "x": 1, "x1": 1, "x2": 1, "x3": 1, "x_1": 1, "x_2": 1, "x_i": 1, "x_n": 1, "xlabel": 1, "xn": 1, "xp": 1, "xpx": 1, "xubo": 1, "xx": 1, "y": 1, "y1": 1, "y2": 1, "y3": 1, "y_1": 1, "y_2": 1, "y_j": 1, "y_m": 1, "ylabel": 1, "you": 0, "your": 0, "ypy": 1, "yy": 1, "z": 1, "zero": 1}, "titles": ["Parth Ganeriwala Comprehensive Breadth Examination", "Tutorial: Coherent Point Drift (CPD) Algorithm for Point Set Registration"], "titleterms": {"": 1, "05": 1, "14": 1, "2017": 1, "In": 1, "It": 1, "On": 1, "The": 1, "abil": 1, "account": 1, "algorithm": 1, "align": 1, "ambigu": 1, "applic": 1, "ar": 1, "assumpt": 1, "background": 1, "base": 1, "between": 1, "breadth": 0, "can": 1, "care": 1, "challeng": 1, "chang": 1, "choic": 1, "closest": 1, "cloud": 1, "code": 1, "coher": 1, "compar": 1, "complex": 1, "comprehens": 0, "comput": 1, "computation": 1, "converg": 1, "correspond": 1, "cpd": 1, "dataset": 1, "defin": 1, "deform": 1, "depend": 1, "desir": 1, "drift": 1, "effect": 1, "effici": 1, "especi": 1, "examin": 0, "exampl": 1, "excel": 1, "expect": 1, "framework": 1, "from": 1, "ganeriwala": 0, "gaussian": 1, "github": 1, "gmm": 1, "hand": 1, "handl": 1, "howev": 1, "http": 1, "i": 1, "icp": 1, "ideal": 1, "includ": 1, "increas": 1, "insight": 1, "involv": 1, "io": 1, "iter": 1, "its": 1, "larg": 1, "level": 1, "limit": 1, "mai": 1, "make": 1, "maxim": 1, "method": 1, "miss": 1, "mixtur": 1, "model": 1, "more": 1, "morpholog": 1, "natur": 1, "nevertheless": 1, "nois": 1, "non": 1, "object": 1, "offer": 1, "optim": 1, "other": 1, "outlier": 1, "overview": 1, "paramet": 1, "parth": 0, "partial": 1, "perform": 1, "point": 1, "presenc": 1, "probabilist": 1, "problem": 1, "provid": 1, "quickli": 1, "real": 1, "registr": 1, "requir": 1, "resourc": 1, "respect": 1, "rigid": 1, "robust": 1, "scenario": 1, "set": 1, "shape": 1, "siavashk": 1, "signific": 1, "simpler": 1, "simplic": 1, "situat": 1, "soft": 1, "specif": 1, "step": 1, "straightforward": 1, "strength": 1, "suffer": 1, "suitabl": 1, "summari": 1, "task": 1, "thank": 1, "time": 1, "transform": 1, "tune": 1, "tutori": 1, "undergo": 1, "variat": 1, "versatil": 1, "weak": 1, "well": 1, "where": 1, "while": 1}})