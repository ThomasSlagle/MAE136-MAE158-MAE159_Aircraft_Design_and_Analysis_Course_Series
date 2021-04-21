import numpy as np
from matplotlib import pyplot as plt

### Fixed aspect ratio curves, varying sweep ###
#2 aisles, 8 abreast, 2 engines

ar_array = np.arange(6,10.5,.5)
sweep_array = np.arange(10, 41, 1)

## non stop ##

array_docpm_ar_nonstop = [np.array([0.01751076, 0.01730059, 0.01723063, 0.01714467, 0.01706502,
       0.01688921, 0.01681569, 0.01672893, 0.01666739, 0.01653383,
       0.01646688, 0.01639463, 0.01623779, 0.01616374, 0.01602473,
       0.01598032, 0.01587387, 0.01584454, 0.01580792, 0.0156964 ,
       0.01568727, 0.01559912, 0.01556169, 0.0154895 , 0.01547987,
       0.01548923, 0.01541061, 0.01541745, 0.01537754, 0.01540184,
       0.01543574]), np.array([0.01739765, 0.01720441, 0.01710604, 0.01701462, 0.01695232,
       0.0167713 , 0.01671376, 0.01664119, 0.016458  , 0.01639955,
       0.01632864, 0.01616034, 0.01609218, 0.01603066, 0.01590496,
       0.01585752, 0.0157346 , 0.01568879, 0.01558283, 0.01555217,
       0.01554255, 0.01545452, 0.01544044, 0.0153439 , 0.01534555,
       0.01526749, 0.01528779, 0.01523013, 0.01524577, 0.0152711 ,
       0.015232  ]), np.array([0.01731408, 0.01719989, 0.01709371, 0.0170198 , 0.01682244,
       0.01678588, 0.0166558 , 0.01652206, 0.01643102, 0.01636816,
       0.01619752, 0.01611715, 0.01604405, 0.01588955, 0.01583228,
       0.01571289, 0.01567273, 0.01555654, 0.01551609, 0.01544652,
       0.01542147, 0.01539831, 0.01530451, 0.01528466, 0.01520953,
       0.01520724, 0.01514153, 0.0151713 , 0.01512433, 0.0151504 ,
       0.01511322]), np.array([0.01732259, 0.01719984, 0.01711283, 0.01700651, 0.01685136,
       0.01673708, 0.01664671, 0.01658719, 0.01641097, 0.0163435 ,
       0.01616637, 0.01608052, 0.01600271, 0.01584301, 0.01578212,
       0.01567738, 0.01561733, 0.01549864, 0.01547171, 0.01537116,
       0.01537499, 0.01533542, 0.01525453, 0.01521957, 0.01517049,
       0.0151674 , 0.01508885, 0.01510798, 0.01506225, 0.01508974,
       0.01507622]), np.array([0.01745532, 0.01732106, 0.01722598, 0.01699642, 0.01691913,
       0.0168225 , 0.01661841, 0.01655509, 0.01645035, 0.01630162,
       0.01623785, 0.01614468, 0.0159668 , 0.01591272, 0.01573746,
       0.01568108, 0.01558524, 0.01553031, 0.01543448, 0.01541366,
       0.0153691 , 0.0152933 , 0.01518194, 0.01517416, 0.01517524,
       0.01510871, 0.01513155, 0.01506249, 0.01509245, 0.01505796,
       0.01509792]), np.array([0.01753102, 0.01741932, 0.01719666, 0.01707416, 0.01702051,
       0.01691568, 0.01672743, 0.01663158, 0.01654448, 0.01636319,
       0.01627048, 0.01619444, 0.01603047, 0.01595035, 0.0157877 ,
       0.01574675, 0.01559001, 0.01555043, 0.01543461, 0.0153956 ,
       0.01538296, 0.0153553 , 0.01532113, 0.01531291, 0.01531427,
       0.01532572, 0.01532207, 0.01534392, 0.01536346, 0.01540888,
       0.01545378]), np.array([0.01767939, 0.01752223, 0.01744572, 0.01718816, 0.01709787,
       0.01701634, 0.01684202, 0.01662936, 0.01653808, 0.016456  ,
       0.01638281, 0.01627631, 0.01618022, 0.01609436, 0.01604078,
       0.01597391, 0.01592003, 0.01587572, 0.01582099, 0.01579654,
       0.01578176, 0.01574998, 0.01572892, 0.01571856, 0.01571932,
       0.01571471, 0.01572619, 0.01574997, 0.01577122, 0.0158217 ,
       0.01587163]), np.array([0.01822195, 0.01795911, 0.01779472, 0.01764387, 0.01754176,
       0.01744975, 0.01732048, 0.01720277, 0.01709623, 0.01700054,
       0.01688573, 0.01679198, 0.01668095, 0.01658189, 0.01649429,
       0.01644294, 0.01638094, 0.01632997, 0.01629003, 0.01623889,
       0.01622166, 0.01618504, 0.01616047, 0.01614824, 0.01614854,
       0.01616157, 0.01617351, 0.01618162, 0.01622255, 0.0162788 ,
       0.01633457]), np.array([0.01901801, 0.01870468, 0.01850984, 0.01833182, 0.0182116 ,
       0.01810341, 0.01795208, 0.01781468, 0.01769071, 0.01757954,
       0.01748076, 0.01733826, 0.01724205, 0.01712701, 0.01702549,
       0.01693727, 0.01686637, 0.01683498, 0.01678869, 0.01672968,
       0.01670935, 0.01669092, 0.01666162, 0.01664638, 0.01664539,
       0.01663749, 0.01665052, 0.01667935, 0.01672487, 0.01676831,
       0.01685039])]
array_doctm_ar_nonstop = [np.array([0.12433677, 0.12284442, 0.12234767, 0.12173728, 0.12117173,
       0.11992342, 0.11940134, 0.1187853 , 0.11834836, 0.11739994,
       0.11692461, 0.1164116 , 0.1152979 , 0.11477215, 0.11378503,
       0.11346976, 0.11271389, 0.1125056 , 0.11224561, 0.11145371,
       0.1113889 , 0.11076297, 0.11049721, 0.10998464, 0.10991626,
       0.10998267, 0.10942443, 0.10947298, 0.10918965, 0.10936216,
       0.10960285]), np.array([0.12353361, 0.12216147, 0.121463  , 0.12081385, 0.12037153,
       0.11908616, 0.11867761, 0.11816231, 0.11686156, 0.11644649,
       0.11594303, 0.11474796, 0.11426401, 0.1138272 , 0.11293461,
       0.11259778, 0.11172499, 0.11139973, 0.11064732, 0.11042959,
       0.11036128, 0.10973625, 0.10963628, 0.10895079, 0.10896246,
       0.10840819, 0.10855234, 0.10814296, 0.108254  , 0.10843381,
       0.10815624]), np.array([0.1229402 , 0.12212942, 0.12137546, 0.12085065, 0.11944929,
       0.11918967, 0.11826606, 0.11731642, 0.11666995, 0.11622361,
       0.11501199, 0.11444129, 0.11392226, 0.11282519, 0.11241858,
       0.11157082, 0.11128566, 0.11046065, 0.11017345, 0.10967942,
       0.10950154, 0.10933709, 0.10867107, 0.10853012, 0.10799667,
       0.10798037, 0.10751386, 0.1077252 , 0.10739171, 0.10757683,
       0.10731283]), np.array([0.12300063, 0.12212902, 0.12151121, 0.1207563 , 0.11965463,
       0.11884314, 0.1182015 , 0.11777888, 0.11652759, 0.11604853,
       0.1147908 , 0.1141812 , 0.1136287 , 0.11249472, 0.11206242,
       0.11131867, 0.11089228, 0.11004954, 0.1098583 , 0.10914434,
       0.10917152, 0.10889055, 0.1083162 , 0.10806797, 0.10771946,
       0.1076975 , 0.10713978, 0.10727557, 0.10695086, 0.10714608,
       0.10705005]), np.array([0.12394313, 0.12298977, 0.12231462, 0.12068463, 0.12013585,
       0.11944972, 0.11800052, 0.11755093, 0.11680725, 0.11575114,
       0.11529834, 0.11463678, 0.11337375, 0.11298975, 0.1117453 ,
       0.11134492, 0.11066445, 0.11027439, 0.10959394, 0.10944609,
       0.10912972, 0.1085915 , 0.10780076, 0.10774553, 0.10775315,
       0.10728077, 0.10744294, 0.10695261, 0.10716534, 0.1069204 ,
       0.10720419]), np.array([0.12448063, 0.1236875 , 0.12210646, 0.12123666, 0.12085572,
       0.12011137, 0.11877465, 0.11809403, 0.11747559, 0.11618833,
       0.11553002, 0.11499011, 0.11382583, 0.11325691, 0.11210203,
       0.11181127, 0.11069829, 0.11041727, 0.10959489, 0.10931785,
       0.10922815, 0.10903171, 0.10878906, 0.10873071, 0.10874035,
       0.10882168, 0.10879574, 0.10895089, 0.10908965, 0.10941218,
       0.109731  ]), np.array([0.12553411, 0.12441817, 0.12387496, 0.1220461 , 0.12140502,
       0.12082608, 0.11958829, 0.11807833, 0.11743014, 0.11684731,
       0.11632765, 0.11557142, 0.11488917, 0.11427946, 0.11389906,
       0.11342423, 0.1130416 , 0.11272703, 0.11233838, 0.11216477,
       0.11205986, 0.11183418, 0.11168461, 0.11161108, 0.11161648,
       0.11158373, 0.11166528, 0.11183411, 0.111985  , 0.11234342,
       0.11269795]), np.array([0.12938661, 0.12752032, 0.12635307, 0.12528189, 0.1245569 ,
       0.12390358, 0.12298562, 0.12214984, 0.12139333, 0.12071392,
       0.11989866, 0.119233  , 0.11844458, 0.11774123, 0.11711924,
       0.11675459, 0.11631439, 0.11595248, 0.11566882, 0.11530576,
       0.11518335, 0.11492337, 0.11474889, 0.11466205, 0.11466421,
       0.11475672, 0.1148415 , 0.11489906, 0.11518972, 0.11558909,
       0.11598513]), np.array([0.13503916, 0.13281426, 0.13143085, 0.1301668 , 0.12931316,
       0.12854495, 0.12747041, 0.1264948 , 0.1256145 , 0.12482511,
       0.12412373, 0.12311188, 0.12242874, 0.12161191, 0.12089106,
       0.12026467, 0.11976124, 0.11953834, 0.11920961, 0.11879065,
       0.11864624, 0.11851543, 0.11830738, 0.11819916, 0.11819209,
       0.11813599, 0.11822854, 0.11843323, 0.11875647, 0.1190649 ,
       0.11964776])]
array_w_ar_nonstop = [np.array([668395., 659010., 656250., 652845., 649720., 641890., 639060.,
       635705., 633400., 627410., 624920., 622230., 615370., 612645.,
       606565., 605070., 600350., 599490., 598350., 593480., 593470.,
       589575., 588455., 585235., 585255., 586050., 582615., 583335.,
       581485., 582935., 584790.]), np.array([665810., 657090., 653135., 649490., 647075., 638995., 636815.,
       634030., 625955., 623790., 621130., 613755., 611245., 609020.,
       603460., 601840., 596440., 594915., 590245., 589360., 589340.,
       585445., 585270., 581065., 581550., 578125., 579390., 576820.,
       577925., 579440., 577665.]), np.array([663880., 659215., 654910., 651980., 643145., 641815., 636605.,
       630515., 626975., 624620., 617085., 614045., 611320., 604535.,
       602495., 597200., 595890., 590770., 589475., 586295., 585655.,
       585105., 580985., 580595., 577270., 577620., 574710., 576395.,
       574280., 575845., 574160.]), np.array([666135., 661085., 657570., 653260., 646155., 641560., 637990.,
       635730., 627860., 625305., 617465., 614185., 611255., 604225.,
       602030., 597320., 595190., 589950., 589210., 584755., 585305.,
       584085., 580485., 579480., 577220., 577550., 574110., 575380.,
       573320., 574965., 574265.]), np.array([674090., 668515., 664635., 654295., 651220., 647345., 638200.,
       635775., 631625., 624905., 622515., 618910., 611055., 609120.,
       601435., 599435., 595100., 593190., 588895., 588425., 586985.,
       583570., 578705., 578825., 579325., 576360., 577790., 574750.,
       576505., 574930., 577135.]), np.array([679330., 674695., 664555., 659515., 657435., 653190., 644660.,
       640845., 637420., 629285., 625665., 622765., 615450., 612415.,
       605215., 603855., 596955., 595680., 590535., 589310., 589200.,
       588475., 587490., 587605., 588135., 589105., 589455., 590895.,
       592255., 594730., 597205.]), np.array([687455., 680830., 677715., 666045., 662385., 659120., 651900.,
       642320., 638705., 635500., 632690., 628480., 624730., 621430.,
       619515., 617050., 615150., 613670., 611760., 611150., 610965.,
       610070., 609645., 609690., 610225., 610545., 611570., 613140.,
       614620., 617380., 620140.]), np.array([715050., 703715., 696730., 690365., 686150., 682395., 677040.,
       672210., 667885., 664050., 659405., 655690., 651245., 647335.,
       643935., 642115., 639850., 638075., 636790., 635035., 634760.,
       633660., 633095., 633080., 633625., 634740., 635825., 636760.,
       639150., 642235., 645325.]), np.array([755900., 742130., 733685., 726020., 720945., 716425., 710020.,
       704255., 699105., 694540., 690540., 684635., 680785., 676125.,
       672075., 668625., 665950., 665025., 663460., 661350., 660955.,
       660660., 659900., 659775., 660295., 660525., 661700., 663595.,
       666255., 668845., 673190.])]

array_docpm_ar_onestop = [np.array([0.01650502, 0.01648685, 0.01643946, 0.01638063, 0.01634186,
       0.0162561 , 0.01620857, 0.01616531, 0.01614048, 0.01610576,
       0.01607549, 0.01604406, 0.01600443, 0.01596962, 0.0159519 ,
       0.01592671, 0.01590649, 0.01584071, 0.0158306 , 0.01582579,
       0.0158263 , 0.01582155, 0.01582262, 0.01582949, 0.01584247,
       0.01586172, 0.01582021, 0.01582938, 0.0158637 , 0.01591365,
       0.01595462]), np.array([0.01618609, 0.01613433, 0.01610299, 0.0160593 , 0.01601966,
       0.01598439, 0.01593592, 0.01590658, 0.01579155, 0.01578473,
       0.01574045, 0.01572246, 0.01566955, 0.01565991, 0.01561741,
       0.01561645, 0.01559603, 0.01558088, 0.01550132, 0.01550789,
       0.01550887, 0.01550449, 0.01550603, 0.01551349, 0.01552718,
       0.01553786, 0.01550968, 0.01553748, 0.0155636 , 0.01560606,
       0.01564829]), np.array([0.01594303, 0.01589017, 0.01584179, 0.01581374, 0.01577357,
       0.01573775, 0.01570377, 0.01558507, 0.01557446, 0.01552493,
       0.01552207, 0.01546298, 0.01544935, 0.01542666, 0.01539593,
       0.01538285, 0.01536242, 0.01527953, 0.01527008, 0.01527726,
       0.01527857, 0.0152747 , 0.01527672, 0.01528478, 0.01523369,
       0.01525479, 0.01527673, 0.01531476, 0.0153509 , 0.01539475,
       0.01543821]), np.array([0.01578078, 0.01566717, 0.01561781, 0.01558948, 0.01554876,
       0.01551238, 0.01547814, 0.01546329, 0.015407  , 0.01540017,
       0.01536827, 0.01533539, 0.0152803 , 0.01520239, 0.01518441,
       0.01515892, 0.01515083, 0.01513591, 0.01512637, 0.0151225 ,
       0.01513558, 0.01506684, 0.01506935, 0.01507818, 0.01509346,
       0.0151152 , 0.01513821, 0.01516845, 0.01516036, 0.01519663,
       0.01525022]), np.array([0.0156512 , 0.01559491, 0.01556126, 0.01547418, 0.01541503,
       0.01537761, 0.01535844, 0.01532714, 0.01526944, 0.01526253,
       0.01522989, 0.01519642, 0.01516778, 0.01507652, 0.01504507,
       0.0150319 , 0.01501128, 0.01500841, 0.01499892, 0.01499509,
       0.0149972 , 0.01494099, 0.01494381, 0.01495305, 0.01496903,
       0.01499172, 0.01501568, 0.0150032 , 0.01504182, 0.01507956,
       0.0151351 ]), np.array([0.01555384, 0.01547687, 0.01546031, 0.01539362, 0.01535013,
       0.01534568, 0.01527496, 0.01524296, 0.01523153, 0.01513836,
       0.01510513, 0.01507079, 0.0150417 , 0.01501758, 0.01498493,
       0.01497137, 0.01495011, 0.01488265, 0.01487326, 0.01488177,
       0.01488405, 0.01488075, 0.01489496, 0.01490444, 0.01485866,
       0.01488217, 0.01490695, 0.01493947, 0.01498001, 0.01503838,
       0.01503366]), np.array([0.01550654, 0.01542657, 0.01539041, 0.01534013, 0.01533137,
       0.01527269, 0.01523459, 0.01521824, 0.0151188 , 0.01507899,
       0.0150603 , 0.01502474, 0.01499443, 0.01494061, 0.01493567,
       0.01492159, 0.01484812, 0.01483243, 0.01483551, 0.01483176,
       0.01483449, 0.01484292, 0.01484601, 0.01479392, 0.01481129,
       0.01484655, 0.01487223, 0.01490591, 0.01494802, 0.01492734,
       0.0149873 ]), np.array([0.01544759, 0.01538498, 0.01536732, 0.01531483, 0.01526768,
       0.0152257 , 0.01520448, 0.01515206, 0.01510547, 0.01509802,
       0.01499318, 0.01495696, 0.01492641, 0.01490126, 0.01488163,
       0.01486737, 0.01485894, 0.01485617, 0.01479509, 0.01479153,
       0.01479444, 0.01479108, 0.01479483, 0.01482904, 0.01478463,
       0.01479929, 0.01483711, 0.0148618 , 0.01491618, 0.01491711,
       0.01495977]), np.array([0.01547259, 0.01542812, 0.01538872, 0.01535398, 0.01528466,
       0.01518683, 0.01516485, 0.01511071, 0.01509821, 0.01505534,
       0.0150183 , 0.01499689, 0.01494842, 0.0149378 , 0.01483541,
       0.01482128, 0.01481281, 0.01481043, 0.01481398, 0.01479718,
       0.01481369, 0.01475985, 0.01476392, 0.01477549, 0.01479484,
       0.01482195, 0.0148506 , 0.01488775, 0.01493412, 0.01498998,
       0.01505591])]
array_doctm_ar_onestop = [np.array([0.12301875, 0.12288336, 0.12253015, 0.12209169, 0.12180271,
       0.12116351, 0.12080919, 0.12048676, 0.12030169, 0.12004293,
       0.11981731, 0.11958306, 0.11928766, 0.11902821, 0.11889613,
       0.11870836, 0.11855769, 0.11806739, 0.11799204, 0.11795623,
       0.11795997, 0.11792462, 0.11793254, 0.11798375, 0.11808053,
       0.11822403, 0.1179146 , 0.11798294, 0.11823875, 0.11861107,
       0.11891639]), np.array([0.12064167, 0.12025586, 0.1200223 , 0.11969664, 0.11940123,
       0.11913833, 0.11877706, 0.11855839, 0.11770098, 0.11765017,
       0.11732016, 0.11718605, 0.11679169, 0.11671986, 0.11640306,
       0.11639588, 0.1162437 , 0.11613082, 0.11553781, 0.11558675,
       0.11559408, 0.11556141, 0.11557288, 0.11562851, 0.11573055,
       0.1158101 , 0.11560008, 0.11580727, 0.11600196, 0.11631849,
       0.11663319]), np.array([0.11883004, 0.11843602, 0.11807543, 0.11786642, 0.11756699,
       0.11729999, 0.11704677, 0.116162  , 0.11608293, 0.11571375,
       0.11569245, 0.11525201, 0.11515041, 0.1149813 , 0.11475225,
       0.11465481, 0.1145025 , 0.11388469, 0.11381428, 0.11386774,
       0.11387756, 0.11384869, 0.11386374, 0.11392382, 0.11354304,
       0.11370029, 0.11386382, 0.11414727, 0.11441663, 0.11474351,
       0.11506742]), np.array([0.11762072, 0.11677392, 0.11640604, 0.1161949 , 0.11589138,
       0.1156202 , 0.115365  , 0.11525435, 0.11483479, 0.11478388,
       0.11454615, 0.11430106, 0.11389041, 0.1133097 , 0.11317575,
       0.11298576, 0.11292547, 0.11281424, 0.1127431 , 0.11271426,
       0.11281174, 0.11229945, 0.11231817, 0.11238393, 0.11249783,
       0.11265985, 0.1128314 , 0.11305678, 0.11299646, 0.11326684,
       0.11366623]), np.array([0.11665488, 0.11623535, 0.11598452, 0.11533548, 0.11489461,
       0.11461574, 0.11447285, 0.11423957, 0.11380946, 0.11375797,
       0.11351471, 0.11326526, 0.11305181, 0.11237157, 0.11213717,
       0.112039  , 0.11188534, 0.11186396, 0.11179322, 0.11176463,
       0.11178035, 0.11136145, 0.11138242, 0.11145134, 0.11157037,
       0.1117395 , 0.11191809, 0.11182509, 0.11211296, 0.11239424,
       0.11280823]), np.array([0.11592922, 0.11535552, 0.11523212, 0.11473509, 0.11441093,
       0.11437771, 0.11385063, 0.11361212, 0.11352696, 0.11283253,
       0.11258485, 0.11232883, 0.11211204, 0.11193227, 0.11168889,
       0.11158786, 0.11142941, 0.11092659, 0.11085662, 0.11092001,
       0.11093699, 0.11091243, 0.11101836, 0.11108899, 0.11074779,
       0.11092304, 0.1111077 , 0.11135006, 0.11165227, 0.1120873 ,
       0.11205215]), np.array([0.11557668, 0.11498065, 0.11471113, 0.11433639, 0.11427111,
       0.1138337 , 0.11354975, 0.11342786, 0.11268668, 0.11238997,
       0.11225066, 0.11198562, 0.11175974, 0.11135857, 0.11132174,
       0.11121682, 0.11066919, 0.1105523 , 0.11057522, 0.11054725,
       0.11056765, 0.11063047, 0.11065346, 0.11026527, 0.11039473,
       0.1106575 , 0.11084892, 0.11109996, 0.11141384, 0.11125971,
       0.11170661]), np.array([0.11513734, 0.11467063, 0.11453902, 0.11414781, 0.11379634,
       0.11348351, 0.11332534, 0.11293458, 0.11258736, 0.11253185,
       0.1117504 , 0.11148042, 0.11125278, 0.11106527, 0.11091895,
       0.1108127 , 0.11074986, 0.11072925, 0.11027395, 0.11024742,
       0.11026912, 0.11024411, 0.110272  , 0.11052704, 0.11019603,
       0.11030529, 0.11058718, 0.11077117, 0.11117648, 0.11118341,
       0.11150142]), np.array([0.11532369, 0.11499217, 0.11469856, 0.11443961, 0.11392291,
       0.11319378, 0.11302994, 0.11262643, 0.11253325, 0.11221372,
       0.11193763, 0.11177809, 0.11141681, 0.11133767, 0.1105745 ,
       0.11046915, 0.11040603, 0.11038827, 0.11041476, 0.11028956,
       0.11041259, 0.11001131, 0.11004163, 0.11012786, 0.11027207,
       0.11047416, 0.11068768, 0.11096459, 0.11131019, 0.11172658,
       0.11221799])]
array_w_ar_onestop = [np.array([442265., 441785., 440450., 438785., 437715., 434925., 433610.,
       432425., 431785., 430860., 430070., 429250., 428195., 427285.,
       426880., 426260., 425790., 423660., 423495., 423490., 423645.,
       423650., 423830., 424185., 424725., 425455., 424080., 424520.,
       425710., 427370., 428770.]), np.array([435395., 433920., 433055., 431830., 430730., 429765., 428415.,
       427635., 423990., 423885., 422680., 422255., 420805., 420635.,
       419500., 419595., 419120., 418805., 416265., 416600., 416775.,
       416795., 416995., 417375., 417945., 418430., 417450., 418455.,
       419415., 420870., 422325.]), np.array([430540., 429025., 427650., 426885., 425765., 424780., 423855.,
       420075., 419855., 418485., 418505., 416860., 416570., 416015.,
       415225., 414965., 414490., 411835., 411695., 412055., 412245.,
       412285., 412505., 412910., 411235., 412040., 412875., 414200.,
       415475., 416990., 418500.]), np.array([427990., 424295., 422885., 422110., 420970., 419965., 419030.,
       418680., 417095., 416995., 416150., 415280., 413750., 411210.,
       410800., 410170., 410065., 409760., 409620., 409655., 410205.,
       407980., 408220., 408655., 409290., 410125., 411005., 412110.,
       411750., 413050., 414885.]), np.array([426055., 424420., 423475., 420575., 418875., 417835., 417350.,
       416505., 414870., 414770., 413900., 413010., 412270., 409315.,
       408500., 408240., 407760., 407820., 407685., 407725., 407950.,
       406095., 406350., 406805., 407470., 408345., 409265., 408770.,
       410150., 411510., 413425.]), np.array([424790., 422520., 422090., 420145., 418910., 418865., 416815.,
       415945., 415705., 412650., 411760., 410840., 410085., 409485.,
       408630., 408360., 407860., 405635., 405505., 405925., 406160.,
       406230., 406840., 407310., 405780., 406690., 407645., 408845.,
       410300., 412315., 412080.]), np.array([425155., 422780., 421750., 420295., 420115., 418415., 417350.,
       416955., 413685., 412585., 412135., 411175., 410380., 408870.,
       408860., 408575., 406150., 405825., 406080., 406130., 406385.,
       406820., 407095., 405360., 406085., 407370., 408365., 409615.,
       411135., 410400., 412490.]), np.array([424955., 423100., 422635., 421105., 419745., 418550., 418000.,
       416495., 415175., 415065., 411625., 410640., 409835., 409200.,
       408740., 408450., 408345., 408420., 406375., 406435., 406700.,
       406775., 407075., 408330., 406830., 407485., 408870., 409850.,
       411770., 411715., 413280.]), np.array([427525., 426220., 425080., 424090., 422035., 418755., 418180.,
       416615., 416345., 415140., 414120., 413590., 412225., 412040.,
       408680., 408395., 408290., 408380., 408660., 408310., 409005.,
       407190., 407505., 408060., 408865., 409920., 411030., 412415.,
       414100., 416095., 418420.])]







#Weight v Sweep
fig1, (ax1, ax2) = plt.subplots(1,2, figsize = (16,7))
fig1.subplots_adjust(bottom = 0.2, wspace = 0.4)

ax1.scatter(sweep_array, array_w_ar_nonstop[0], label='AR: 6')
ax1.scatter(sweep_array, array_w_ar_nonstop[1], label='AR: 6.5')
ax1.scatter(sweep_array, array_w_ar_nonstop[2], label='AR: 7')
ax1.scatter(sweep_array, array_w_ar_nonstop[3], label='AR: 7.5')
ax1.scatter(sweep_array, array_w_ar_nonstop[4], label='AR: 8')
ax1.scatter(sweep_array, array_w_ar_nonstop[5], label='AR: 8.5')
ax1.scatter(sweep_array, array_w_ar_nonstop[6], label='AR: 9')
ax1.scatter(sweep_array, array_w_ar_nonstop[7], label='AR: 9.5')
ax1.scatter(sweep_array, array_w_ar_nonstop[8], label='AR: 10')
ax1.set_xlabel('Sweep Angle (Degrees)', fontsize=14)
ax1.set_ylabel('Takeoff Weight (pounds)', fontsize=14)

ax2.scatter(sweep_array, array_w_ar_onestop[0], label='AR: 6')
ax2.scatter(sweep_array, array_w_ar_onestop[1], label='AR: 6.5')
ax2.scatter(sweep_array, array_w_ar_onestop[2], label='AR: 7')
ax2.scatter(sweep_array, array_w_ar_onestop[3], label='AR: 7.5')
ax2.scatter(sweep_array, array_w_ar_onestop[4], label='AR: 8')
ax2.scatter(sweep_array, array_w_ar_onestop[5], label='AR: 8.5')
ax2.scatter(sweep_array, array_w_ar_onestop[6], label='AR: 9')
ax2.scatter(sweep_array, array_w_ar_onestop[7], label='AR: 9.5')
ax2.scatter(sweep_array, array_w_ar_onestop[8], label='AR: 10')
ax2.set_xlabel('Sweep Angle (Degrees)', fontsize=14)
ax2.legend(loc = 'lower center', bbox_to_anchor=(0,0,1,1),bbox_transform = plt.gcf().transFigure, ncol = 3)
ax1.set_title('Non-stop Aircraft', fontsize = 14)
ax2.set_title('One-stop Aircraft', fontsize = 14)
plt.savefig(r'MAE 159\Other data\Images\Weight v Sweep Angle.png')
plt.close()


#DOC TM
fig2, (axx1, axx2) = plt.subplots(1,2, figsize = (16,7))
fig2.subplots_adjust(bottom = 0.2, wspace = 0.4)

axx1.scatter(sweep_array, array_doctm_ar_nonstop[0], label='AR: 6')
axx1.scatter(sweep_array, array_doctm_ar_nonstop[1], label='AR: 6.5')
axx1.scatter(sweep_array, array_doctm_ar_nonstop[2], label='AR: 7')
axx1.scatter(sweep_array, array_doctm_ar_nonstop[3], label='AR: 7.5')
axx1.scatter(sweep_array, array_doctm_ar_nonstop[4], label='AR: 8')
axx1.scatter(sweep_array, array_doctm_ar_nonstop[5], label='AR: 8.5')
axx1.scatter(sweep_array, array_doctm_ar_nonstop[6], label='AR: 9')
axx1.scatter(sweep_array, array_doctm_ar_nonstop[7], label='AR: 9.5')
axx1.scatter(sweep_array, array_doctm_ar_nonstop[8], label='AR: 10')
axx1.set_xlabel('Sweep Angle (Degrees)', fontsize=14)
axx1.set_ylabel('DOC per Ton per Mile ($)', fontsize=14)

axx2.scatter(sweep_array, array_doctm_ar_onestop[0], label='AR: 6')
axx2.scatter(sweep_array, array_doctm_ar_onestop[1], label='AR: 6.5')
axx2.scatter(sweep_array, array_doctm_ar_onestop[2], label='AR: 7')
axx2.scatter(sweep_array, array_doctm_ar_onestop[3], label='AR: 7.5')
axx2.scatter(sweep_array, array_doctm_ar_onestop[4], label='AR: 8')
axx2.scatter(sweep_array, array_doctm_ar_onestop[5], label='AR: 8.5')
axx2.scatter(sweep_array, array_doctm_ar_onestop[6], label='AR: 9')
axx2.scatter(sweep_array, array_doctm_ar_onestop[7], label='AR: 9.5')
axx2.scatter(sweep_array, array_doctm_ar_onestop[8], label='AR: 10')
axx2.set_xlabel('Sweep Angle (Degrees)', fontsize=14)
axx2.legend(loc = 'lower center', bbox_to_anchor=(0,0,1,1),bbox_transform = plt.gcf().transFigure, ncol = 3)
axx1.set_title('Non-stop Aircraft', fontsize = 14)
axx2.set_title('One-stop Aircraft', fontsize = 14)
plt.savefig(r'MAE 159\Other data\Images\DOCTM v Sweep Angle.png')
plt.close()

#AR Plot
fig3, (axxx1, axxx2) = plt.subplots(1,2, figsize = (16,7))
fig3.subplots_adjust(bottom = 0.2, wspace = 0.4)

axxx1.scatter(sweep_array, array_doctm_ar_nonstop[3], label='AR: 7.5')
axxx1.scatter(sweep_array, array_doctm_ar_nonstop[4], label='AR: 8')

ar_nonstop_min3 = min(array_doctm_ar_nonstop[3])
xpos_n3 = np.where(array_doctm_ar_nonstop[3] == ar_nonstop_min3)
sweep_nonstop_min3 = sweep_array[xpos_n3]
axxx1.annotate('AR: 7.5 Minimum', xy = (sweep_nonstop_min3,ar_nonstop_min3), xytext=(sweep_nonstop_min3-5, ar_nonstop_min3+0.005), arrowprops=dict(color='#1f77b4', shrink=0.05), color = '#1f77b4')

ar_nonstop_min4 = min(array_doctm_ar_nonstop[4])
xpos_n4 = np.where(array_doctm_ar_nonstop[4] == ar_nonstop_min4)
sweep_nonstop_min4 = sweep_array[xpos_n4]
axxx1.annotate('AR: 8 Minimum', xy = (sweep_nonstop_min4,ar_nonstop_min4), xytext=(sweep_nonstop_min4-8, ar_nonstop_min4-0.005), arrowprops=dict(color='#ff7f0e', shrink=0.05), color = '#ff7f0e')

ar_nonstop_min4 = min(array_doctm_ar_nonstop[4])

axxx1.set_xlabel('Sweep Angle (Degrees)', fontsize=14)
axxx1.set_ylabel('DOC per Ton per Mile ($)', fontsize=14)

axxx2.scatter(sweep_array, array_doctm_ar_onestop[7], label='AR: 9.5')
axxx2.scatter(sweep_array, array_doctm_ar_onestop[8], label='AR: 10')

ar_onestop_min7 = min(array_doctm_ar_onestop[7])
xpos_n7 = np.where(array_doctm_ar_onestop[7] == ar_onestop_min7)
sweep_onestop_min7 = sweep_array[xpos_n7]
axxx2.annotate('AR: 9.5 Minimum', xy = (sweep_onestop_min7,ar_onestop_min7), xytext=(sweep_onestop_min7-5, ar_onestop_min7+0.005), arrowprops=dict(color='#1f77b4', shrink=0.05), color = '#1f77b4')

ar_onestop_min8 = min(array_doctm_ar_onestop[8])
xpos_n8 = np.where(array_doctm_ar_onestop[8] == ar_onestop_min8)
sweep_onestop_min8 = sweep_array[xpos_n8]
axxx2.annotate('AR: 10 Minimum', xy = (sweep_onestop_min8,ar_onestop_min8), xytext=(sweep_onestop_min8-8, ar_onestop_min8-0.005), arrowprops=dict(color='#ff7f0e', shrink=0.05), color = '#ff7f0e')

ar_onestop_min3 = min(array_doctm_ar_onestop[3])
ar_onestop_min4 = min(array_doctm_ar_onestop[4])

axxx2.set_xlabel('Sweep Angle (Degrees)', fontsize=14)

axxx1.set_title('Non-stop Aircraft', fontsize = 14)
axxx2.set_title('One-stop Aircraft', fontsize = 14)
axxx2.legend(loc = 'lower center', bbox_to_anchor=(0,0,1,1),bbox_transform = plt.gcf().transFigure, ncol = 3)

plt.savefig(r'MAE 159\Other data\Images\AR Plot.png')
plt.close()

print(np.array([sweep_nonstop_min3, sweep_nonstop_min4, sweep_onestop_min7, sweep_onestop_min8]))
print(np.array([min(array_doctm_ar_nonstop[3]), min(array_doctm_ar_nonstop[4]), min(array_doctm_ar_onestop[7]), min(array_doctm_ar_onestop[8])]))

