import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
from collections import namedtuple
import math
# import sympy as sym
import csv

# Penalties
MAX_DEPTH = 5
FOOD_CONSTR_PENALTY = 0
TUNNEL_CONSTR_PENALTY = 0
NEED_DRONE_PENALTY = 30
NEED_SOLDIER_PENALTY = 50
NEED_WORKER_PENALTY = 100

STALEMATE_TURNS = 250
GENERATIONS = 50 # How many times to run through all the training data
SEED = 12345 # The seed for selecting the training data
TRAINING_FILE = None # "./training_all.csv"
WEIGHTS_FILE = None #"./weights_all.txt"
EPSILON = 0.0001 # A small value. Used to do a numerical gradient check.
LEARNING_RATE = 0.1
TRAIN = False # Whether to train using training data on the first turn.
ONLINE_LEARNING = False # Whether to train during every action of a game.
LAYERS = [14, 30, 5, 1] # The sizes of each layer. First number is the number of inputs. 

Features = namedtuple("Features", ["foodCount", "enemyMoreFood",
                "hasDrone", "hasSoldier", "mutlipleWorkers", "hasWorker", "enemyHasWorker",
                "numScaryFighters", "areScaryFighters", "droneDist", "soldierDist", "queenPenalty",
                "workerCost", "roundTripCost"])

# The weights learned after training
LEARNED_WEIGHTS = [
[   # Layer 1: 14 -> 30
[-0.2139007394125279, -0.2522115302143484, -0.6397502057551697, 0.5287397738226786, -0.10268221342549859, -0.3105687559542636, -0.4588389259462934, -0.3756262898157483, 0.2019679503985456, 0.9527644799761008, -0.8156126131003058, -0.07390522591072071, -0.7748742916261926, -0.146218144787211, 0.9963653551025349],
[-4.351140791577438, 0.4678473104759443, 0.1406669494574794, -0.09637938894536349, 1.1583474616855276, -0.4038141286543172, -0.8397097588122233, 0.06720230969737069, -0.33665972442748104, 0.4517214817531189, -0.017630294746068108, -0.026592577448881317, -0.017274246763084037, -0.008781051490354469, 0.15624916705344366],
[0.09804580741487341, -1.0702203330390998, -0.6118200641510291, -0.9409184345419079, 0.5994006217758685, -0.012079308817414928, -0.7369051638726406, -0.2543449119297222, -0.15448952187245327, 0.37414352041374865, 0.2875669899785114, 0.7905501671263195, 0.6061505094186358, 0.6159306377895598, -0.42751777657956],
[0.15599297274534762, 0.4581631067342048, -0.5689691510187523, -0.584299579276669, -0.28109463557482817, 0.4999927019551601, -0.033638381266439595, 0.07619406931259676, 0.0745212519107824, 0.8956120033190421, -0.9963407867199879, -0.7883140303862645, -0.810576831315346, -0.9771629595645598, 0.8489023660618064],
[-1.079755571170916, -0.3623180131290154, -0.03547208172378009, -0.1582024386358358, -0.11551579835024656, -0.03744559583385323, -0.14042833090828755, 0.04436359356357528, -0.3514130341568291, -0.2117215008826448, -0.007990082957505662, 0.009838774642190625, -0.005928452516526114, -0.0004395279593777345, 0.32697586023317393],
[0.08989532030222926, -0.501557313420424, 0.603889231370038, 0.3578682185661396, -0.5754435658731162, 0.7327236230993482, -0.3643773226593496, 0.5846631305220718, 0.8291138128028867, 0.4864900395281296, 0.7785745504608106, -0.44870444101722384, 0.8622970771636103, 0.4204893085876446, 0.7905760033388681],
[0.01738816218496243, -0.5047934998276619, 0.5599709693016472, 0.44699116145382983, 0.8496886726612285, 0.705838801915893, 0.15483975570233222, -0.0072750942908645045, 0.756768816463298, 0.1698355454610479, -0.6282064144707319, 0.02786276119110778, -0.7258974724537438, 0.6482784563942294, 0.19187823501006768],
[0.13276897209413077, 0.7154930608078336, -0.011944864571934426, 0.539648037426233, 0.6354552818416676, -0.579193806299877, -0.3321433115894677, 0.6899601578692924, -0.7622883700825712, -0.23899909378643874, 0.2615471094455997, 0.6774194814778304, 0.7577175540157536, -0.6472867632216226, 0.50238711264496],
[2.9545613355162765, -0.4994533461622067, 0.6326539331831285, 0.2869701343467058, -1.4510477716284809, 0.786909262968958, 1.0855914886849192, 0.17340036076083187, 0.07196027807744605, -0.3911576689621718, -0.20911639827789205, -0.15997094830169806, -0.5422293815845506, -0.0024704451516288986, -0.2111319922031736],
[0.6723242167067317, 0.6366947641151359, -0.5347109454399415, 0.6399617381261425, 0.8844773898100116, -0.4337261319142449, -0.0830740560606653, -0.6258851167055542, -0.06525634610434648, 0.9883523252803043, 0.6812850180279485, -0.1222043758773459, 0.45173990468050634, -0.5498500226657793, 0.6920743931675951],
[1.351454341398352, 0.19476138734434748, -0.04534816982757214, 0.01530400907894639, -4.630753117896875, -0.48135140065257487, 2.1175645994735754, 0.054565055304818755, -2.929849345356954, -3.867281810710715, -0.024107032633962182, -0.02749021253195349, -0.021393452602193042, -0.012249911580383232, -0.13082400184824983],
[-0.7650398072086201, -0.6150448719913243, -0.8098869112083352, -0.7229226759393499, -0.9720048739519408, -0.37641464362278787, 0.269992562974713, 0.27741218766206327, 0.4986691673249392, -0.20098217648704422, 0.12731616903416224, -0.5453940252669834, -0.5605574055586491, -0.3930791491213207, 0.5725354685609659],
[-0.4196737550909327, -0.0510655563762705, 0.06566500512450325, 0.15666524858528338, 0.8438876730096064, -0.6285228729972112, 0.7277103757553377, -0.7345217448616681, 0.7273992292915384, -0.061010768319768725, 0.5021068476793467, -0.9190743408994176, -1.0642668672997342, 0.6825347509505425, 0.7626239773867044],
[0.25769108588226586, 0.02421324938824083, 0.8848691543630902, -0.07666899396618841, 0.27543771400520684, -0.8412881324118748, 0.5205709576700335, -0.1264046143982739, -0.5540873073308965, -0.7784134591631586, 0.382257415195733, -0.802418062991935, -1.0157493978576864, 0.6588186781322619, 0.7811162881618454],
[-4.411901426450588, -0.09515352506839067, 3.895827437029692, -4.126938118517, -0.3527739871908883, -0.05698715280617532, -3.6480775602569833, 3.515067508580242, 0.5671590559669235, 0.4186881744347644, 0.010845343084099311, -0.015237633799382293, 0.008417397470437253, 0.007822622941698143, 0.1646738728996409],
[0.7304278523664275, 0.01182535225608081, -0.23383859581698163, 0.25707218401255993, 0.37592368784948765, -0.4189500248703169, 0.11276178136163863, 0.9271944899773693, 0.5289100313004397, -1.0119216492807126, -0.025997157543830366, -1.0771608838561635, 0.9657164751976212, -0.4016609430586986, 0.2744973948207085],
[0.1986878675642362, 0.13083740687856366, -0.5460892685586414, 0.621139835073799, -0.6677078748856674, 0.5741133862480926, 0.2781575355888344, 0.5985431628377823, -0.9899836011820863, -0.023922998220905614, -0.8421740148034731, 0.09405557659316455, -0.5707395639941607, -0.3580531930594104, 0.35779820946991914],
[0.9454444462477541, -0.37875163268204126, -0.7567658507013438, -0.19734008934115105, -0.7235999789527964, 0.9392119946558037, 0.1790559931883868, 0.3476014269390233, -0.28252499811843296, 0.36202720313239417, -0.742148804383238, 0.7716719128838269, -0.893808472476658, -0.6538411543923566, -0.6144014979633036],
[0.47964333197719894, -0.17133235670497912, 0.1265780659043389, -0.09805683249116931, 0.6279187344339497, -0.3939627920058241, 1.6672476188471352, 0.09333515450402816, 0.23552682662961097, 0.058453174554422546, -0.02102627808131681, -0.01876994798494112, -0.020355472187414597, -0.015662446473943874, -0.4437653461367542],
[0.9698405611608134, 0.8003942000963294, -0.47720495667880336, -0.4610232499085952, 0.24485851603098213, -0.666439916625604, 0.04384964579082347, 0.2713119410070708, -0.4533828882617793, -0.5056773259799464, -0.26424951199765023, 0.15417128063576627, 0.5902821447059249, 0.16041642122905617, -0.34423868291288096],
[-0.23125361634328606, -0.264042245831064, -0.6665661988295928, 0.22358591800458796, 0.9919153096400961, 0.5849619267396109, 0.2725144254547077, 0.892920054574939, -0.3179157251767343, 0.23028461651938406, -0.4089294932341509, -0.0827197420371186, -0.49573624403551914, -0.6889676410405904, 0.14808428402682652],
[0.6014703208352952, 0.10264183486159602, -0.407710522774069, 0.5917316257555761, 0.4819746032707021, 0.4772431344126605, -0.054885566257391036, -0.7337041491762542, 0.39124737904779705, 0.6128706569989669, -1.077671958590915, 0.5822132249791037, -0.580681327569055, 0.17753260277912042, -0.4812469841154009],
[0.3217711432761534, -0.48556184293947036, 0.637265460870113, -0.5373560535986236, -0.19196219111023796, -0.03869677897057584, -0.6785447214925419, 0.9151062333792499, -0.6227398426012819, -0.9171328191505075, -0.6693007241718127, 0.2874257913298066, -0.8364310093979753, -0.5187268994573438, 0.4351828475693044],
[-0.21051805773259435, -0.30624852615410897, -0.28114148383743504, 0.6013196151632996, -2.139276420537177, 0.06113675656108843, -2.9475444735986005, -0.14517362920769764, 0.27878082976154406, 0.9009271723002774, 0.007669977618011673, -0.017934793317803035, 0.018054356925573254, 0.015749975060414954, 0.2945883869849404],
[0.28924339757195516, 0.48552586196033937, -0.6137181007431971, -0.896568525073682, 0.8380855188478638, 0.5258641795626925, -0.816858346767107, 0.18110717068667412, 0.028491886547470573, 0.006146715204509806, -0.6210020767214667, -0.9673633413034401, -0.4458313449322377, -0.4485704495234558, -0.5628544287779511],
[1.6467033913135851, -0.5445401100769965, -0.19251765759070355, 0.19640668612159656, -1.5182362086743155, 0.6183168343762744, 1.0663522136273706, 0.021625328123238627, 0.6666489099528667, -0.35746084014837204, -0.10043574463246131, -0.15119207170397758, -0.28675755063648256, -0.08078198560378197, 0.30219264763312836],
[0.5537973543401982, 0.14348393137457058, 0.5733789534441543, -0.5529597343502335, -0.24664194094039887, 0.20348621022947055, 0.7395498500466441, 0.6618586869241201, -0.30963636805915346, -0.309789702324514, 0.742249078646185, -0.009142674682046667, 0.8363119736828689, 0.11131981386556267, 0.36554306652796537],
[-0.15314599260023262, -0.5596188715503574, 0.2321082789938297, 0.5187694647762948, 0.38015348834025486, 0.6543487749025182, 0.6056633210834397, -0.32765103146377217, 0.4011554942299783, -0.34743983407934376, 0.46144956822364275, -0.45659879117669133, 0.7532768311341281, 0.7876944979368512, -0.07794998266846738],
[-0.5064756306206265, -0.6289652719212273, 0.2624482966309471, 0.9257301201105073, -0.1636847759026343, 0.5813334717586822, 0.7997452147519414, 0.11221679929410623, -0.11895256682899923, 0.34091103662979627, -0.40579911713277206, 0.0014497431028444926, -0.4704526312040837, 0.4136204921212189, 0.11830533651485234],
[0.11685007894222373, 0.20609769029691244, -0.2503042918729675, 0.10671538557070775, 1.925035813171048, -0.767552007876052, 1.2569232191408128, -0.12307300024102236, -0.10884353988579393, 9.498407490010601e-05, -0.03841144671720772, -0.033004559578665923, -0.03913829677386174, -0.04328105921730247, -0.15056766592912055]
], [ # Layer 2: 30 -> 5
[-0.09829256505475903, 0.5209878856780429, 0.22328961498197342, -0.41653100958143785, -0.03315655812179338, -0.2255434189786476, 0.1443160258030774, -0.6483554091639354, -0.030416088785322773, -0.7387460765363053, -0.4076162557702902, -0.4470381511521347, -0.5152829194972636, -0.8028598424053319, -0.022976089784584607, -0.37833389361103836, -0.7318735156586701, 0.11678579270189444, 1.0382123860683656, -1.6694503838643224, -0.2615436440942576, -0.3456071896859967, 0.06676844322854229, -0.7180240021070902, 1.3014407331731408, -0.39386349366137763, 0.5964849781401993, -0.6465994718684432, 0.47190967978554144, -0.2976301646127013, -1.328454237952212],
[-0.14937087359960904, -0.17483898026423889, 1.866727928269145, -1.2125237310326327, 0.4300333570011915, -0.2024270907546764, 0.2983872620018874, 0.344794092701219, 0.19306188579896028, -1.1862578556406955, -0.6694108777203303, 1.2095159539200473, 0.3469714477757668, 0.018200822197661143, -0.18371931365711863, -1.2421295993633052, 0.04162087627324364, -0.8849344092165529, -0.14544223592632022, 1.145889992658379, -0.9412075435074732, 0.8481857521305305, -0.010015463598264393, -0.959055630113964, 0.4709020227092116, 0.5140584843403913, -0.9543455554700966, 0.007294884958347047, -0.0538649608567732, -0.9009928300169128, 0.06510119834282668],
[-1.0417642770311644, -0.8646273062889451, 0.34867571876425746, -0.05372639223857991, -0.9487711555171143, -1.7412492160109283, 0.10506051643158185, -0.4049565933160582, -0.9538057288633564, -1.041951942685846, -0.9239195050826838, 0.22188136002256975, 0.628594981422203, -0.5162185286342804, 0.6928744917022341, -1.18999586856141, 0.010597267921454127, -0.6910864667756172, -0.4995021824877179, 0.7304644070490536, -0.28622766817956763, -0.08282113160010574, -0.15223978339410693, -0.27307997582625027, -1.9948141902446346, 0.24141081766999378, -0.6916585992851558, 0.626546132974784, 0.5691534583811956, -0.8463557529416049, 1.2731366730606495],
[-1.2415118565664967, -0.2942161284354461, -1.959820381676507, -0.4012589718569578, 0.10040342259977679, 1.850627335663466, 0.49963159955931985, -0.7382150371726628, 0.4503151013152119, 0.9169055928908769, -0.07048434536783756, -0.4443669910331744, -0.4155279291865416, -0.3977682890723976, 0.6117482235290101, 0.7879865287742653, 0.17296736260365012, -1.0137794684749906, -0.41977577915853176, -2.0965299412178977, -1.023989186566998, -0.653779032066378, 0.38419025510263605, 0.01174201497983393, 0.6726906248824369, -0.6427892057684655, 0.36440204428532136, -1.0391733247011488, -0.09437915844773788, 0.5345446188356724, -0.7713612678847053],
[-0.37221682793791494, -0.4594928347755444, 0.5071033034045344, -0.6871697024532064, 0.6991358481428179, -0.24664227834610863, -0.5312425127719667, -0.1305322408123696, 0.24728260832439, -0.7701908128371081, -0.7819660393064226, -1.071998504061109, -0.5056766554753772, 0.3171347143975897, -0.7243608125795417, 0.03571127457840283, 0.7917622378574759, 0.3311870519390004, -0.09714778675761648, -1.0486006951150515, -0.08937090581967111, 0.06089525361284848, -0.14672796106269093, 0.12287685600825494, 0.5451814916173826, 0.09248736223225738, 0.4933830480588479, 0.4135981248837588, -0.08458439971838283, 0.7191113045283063, -0.2903108900742732]
], [ # Layer 3: 5 -> 1 (Output layer)
[0.7072748655422558, 1.3949529538660326, -2.683636479979103, -3.0384677817759256, 2.5938440125117097, 0.8210972724925129]
]]

##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "HW5")
        self.myFood = None
        self.isFirstTurn = None
        self.myConstr = None
        self.foodDist = None
        self.enemyFoodDist = None
        self.bestFoodConstr = None
        self.bestFood = None
        self.network = AIPlayer.NeuralNetwork(LAYERS, LEARNING_RATE, WEIGHTS_FILE, True)
        self.gamesPlayed = 0

    # extractFeatures
    # Descrition: Converts a game state into a Features tuple.
    #
    # Parameters:
    #   currentState: The game state to extract features from.
    #
    # Return:
    #   A Features tuple
    def extractFeatures(self, currentState):
        # Grab needed variables
        me = currentState.whoseTurn
        inventory = getCurrPlayerInventory(currentState)
        enemyInv = getEnemyInv(None, currentState)
        queen = inventory.getQueen()
        drones = getAntList(currentState, me, (DRONE,))
        soldiers = getAntList(currentState, me, (SOLDIER,))
        workers = getAntList(currentState, me, (WORKER,))
        enemyWorkers = getAntList(currentState, 1 - me, (WORKER,))
        enemyFighters = getAntList(currentState, 1 - me, (DRONE, SOLDIER, R_SOLDIER))
        scaryFighters = list(filter(lambda fighter: fighter.coords[1] < 5, enemyFighters))
        anthill = inventory.getAnthill()
        droneSource = drones[0].coords if len(drones) > 0 else anthill.coords
        soldierSource = soldiers[0].coords if len(soldiers) > 0 else anthill.coords

        # Compute features
        food = inventory.foodCount
        enemyFoodCount = enemyInv.foodCount
        enemyMoreFood = 1 if enemyFoodCount >= food else 0
        numDrones = len(drones)
        hasDrone = 1 if numDrones > 0 else 0
        numSoldiers = len(soldiers)
        hasSoldier = 1 if numSoldiers > 0 else 0
        numEnemyWorkers = len(enemyWorkers)
        enemyHasWorker = 1 if numEnemyWorkers > 0 else 0
        numScaryFighters = len(scaryFighters)
        areScaryFighters = 1 if numScaryFighters > 0 else 0
        numWorkers = len(workers)
        hasWorker = 1 if numWorkers > 0 else 0
        multipleWorkers = 1 if numWorkers > 1 else 0

        # Deal with distance features
        droneDist = 0
        if len(enemyWorkers) > 0:
            droneDist += sum(map(lambda enemyWorker: \
                                      self.movesToReach(currentState, droneSource, enemyWorker.coords, DRONE), enemyWorkers))
        droneDist += self.movesToReach(currentState, droneSource, enemyInv.getAnthill().coords, DRONE)

        soldierDist = 0
        soldierDist += sum(
            map(lambda target: self.movesToReach(currentState, soldierSource, target.coords, SOLDIER), scaryFighters))
        soldierDist += self.movesToReach(currentState, soldierSource, enemyInv.getAnthill().coords, SOLDIER)

        queenPenalty = 120.0 / (approxDist(queen.coords, self.bestFoodConstr.coords) + 1) + 120.0 / (
                approxDist(queen.coords, self.bestFood.coords) + 1)

        workerCost = 0
        workerCoords = workers[0].coords if hasWorker else anthill.coords
        workerCarrying = workers[0].carrying if hasWorker else False
        workerCost += self.getWorkerPenalty(currentState, workerCoords)
        workerCost += self.getWorkerCost(currentState, workerCoords, workerCarrying)
        roundTripCost = self.getWorkerCost(currentState, anthill.coords, False)

        # ["foodCount", "enemyMoreFood",
        #         "hasDrone", "hasSoldier", "mutlipleWorkers", "hasWorker", "enemyHasWorker",
        #         "numScaryFighters", "areScaryFighters", "droneDist", "soldierDist", "queenPenalty",
        #         "workerCost", "roundTripCost"]

        return Features(food, enemyMoreFood, hasDrone,
            hasSoldier, multipleWorkers, hasWorker, 
            enemyHasWorker, numScaryFighters, areScaryFighters, 
            droneDist, soldierDist, queenPenalty, 
            workerCost, roundTripCost)

    ## Normalizes features to be between 0 and 1
    #   In refactor, we decided to stop normalizing features (most became binary vars anyway),
    #       so this function currently does nothing.
    def normalizeFeatures(self, features):
        return features

    # train
    # Description: Runs training data through network to train it.
    #
    # Parameters:
    #   trainingPath: A path to a file containing training data.
    #           This file should be a CSV containing all features followed by the heuristic output.
    #   network: A NeuralNetwork to train
    #
    # Return:
    #   none
    #
    # Side Effects:
    #   Will train network.
    #
    # WARNING: Is very slow. Set GENERATIONS to a reasonable number.
    def train(self, trainingPath, network):
        print("Layers: ", end='')
        print(network.layerSizes)

        print("Reading training data")
        trainingData = []
        with open(trainingPath, newline="") as file:
            reader = csv.reader(file)
            for line in reader:
                # One line in a 4 million line file was broken and the file was too big for me to fix, so instead we have this hack
                if len(line) != 15:
                    print("Ignoring line:")
                    print(line)
                    print("---")
                    continue
                row = list(map(lambda x: float(x), line))
                trainingData.append(row)

        # Split into training and testing data
        random.seed(SEED)
        random.shuffle(trainingData)
        random.seed() # Reset seed if used later
        size = len(trainingData)
        cutoff = size * 90 // 100 # 90% of data for training, 10% for testing
        testData = trainingData[:cutoff]
        trainingData = trainingData[cutoff:]

        print("Training")
        for i in range(GENERATIONS):
            print("----- Generation: %d -----" % (i + 1))
            random.shuffle(trainingData)

            for data, learning, label in ((trainingData, True, "Training"), (testData, False, "Test")):
                totalSquareError = 0
                for state in data:
                    expected = state[-1] # The last entry is the heuristic, so we need to extract it
                    normalizedExpected = normalizeHeuristic(expected)
                    rawFeatures = state[:-1]
                    normalizedFeatures = self.normalizeFeatures(rawFeatures)
                    actual = network.eval(normalizedFeatures)[0]
                    if learning:
                        network.adjustWeights([normalizedExpected])
                    error = normalizedExpected - actual
                    totalSquareError += error ** 2
                mse = totalSquareError / len(data)
                print("%s MSE: %f" % (label, mse))

            print("Weights:")
            network.printWeights()
            print()

    # NeuralNetwork
    # Description: A neural net capable of backpropogation
    #
    # Variables:
    #   learningRate: The learning rate parameter for backpropogation.
    #   layerSizes: A list of integers corresponding to the size of each layer, with the first entry being the number of inpupts.
    #   layers: A list of Layer objects.
    #   output: The last output from when eval() was called.
    class NeuralNetwork:

        # init
        # Description: Creates a NeuralNetwork
        #
        # Parameters:
        #   layerSizes: A list of integers corresponding to the size of each layer, with the first entry being the number of inpupts.
        #   learningRate: The learning rate parameter for backpropogation.
        #   weightPath: The path to a weights file. If None, random weights are used.
        #       The weights file is the same format as what printWeights() displays.
        #   useLearnedWeights: If True, weightPath is ignored and LEARNED_WEIGHTS is used for weights.
        def __init__(self, layerSizes, learningRate, weightPath=None, useLearnedWeights=False):
            self.learningRate = learningRate
            self.layerSizes = layerSizes
            self.layers = []
            if weightPath:
                allWeights = self.readWeights(weightPath, layerSizes)
            for i in range(1, len(layerSizes)):
                weights = []
                if useLearnedWeights:
                    weights = LEARNED_WEIGHTS[i-1]
                elif weightPath:
                    weights = allWeights[i-1]
                else:
                    weights = randomMatrix(layerSizes[i-1]+1, layerSizes[i]) # Adding 1 for bias
                
                self.layers.append(AIPlayer.Layer(layerSizes[i], weights, learningRate, sigmoid, sigmoidPrime))

        # readWeights
        # Description: Reads weights from a file. Extra lines at the end of the file are ignored.
        #   The file is not checked for correctness. Make sure the layerSizes used is the same as 
        #   when the weights were created.
        #
        # Parameters:
        #   path: The path to the weights file.
        #   layerSizes: A list of integers corresponding to the size of each layer, with the first entry being the number of inpupts.
        #
        # Return: A list of weight matricies, one layer per entry.
        def readWeights(self, path, layerSizes):
            weights = []
            layer = []
            layerIndex = 1
            with open(path, "r") as weightFile:
                for line in weightFile:
                    values = line[1:-2] # Removes square brackets
                    layer.append(list(map(float, values.split(', '))))
                    if len(layer) == layerSizes[layerIndex]:
                        weights.append(layer)
                        layer = []
                        layerIndex += 1
                        if(layerIndex == len(layerSizes)):
                            break
            return(weights)

        ## Given a tuple of normalized features, compute the network output
        def eval(self, features):
            nextLayerIn = features
            for layer in self.layers:
                nextLayerIn = layer.eval(nextLayerIn)
            self.output = nextLayerIn
            return nextLayerIn

        # Using the last evaluated output and the expected output, adjust the weights
        # Warning: will fail if eval not first called.
        def adjustWeights(self, expected):
            adjustments = []
            nextLayerError = [(expect - out) for out, expect in zip(self.output, expected)]
            for layer in reversed(self.layers):
                error = layer.getErrorTerms(nextLayerError)
                adjustment = layer.adjustWeights(nextLayerError)
                adjustments.append(adjustment)
                nextLayerError = error
            return reversed(adjustments)

        ## Computes the partial derivative of the error function with respect to every weight.
        #  For test purposes.
        #  Return: A array of derivative matricies in the same format as weights.
        def getDerivative(self, features, expected):
            derivatives = []
            for layer in self.layers:
                layerDerivative = []
                for row in layer.weights:
                    rowDerivative = []
                    for i in range(len(row)):
                        originalWeight = row[i]
                        row[i] = originalWeight + EPSILON
                        outHigh = self.eval(features)[0]
                        errHigh = 0.5 * (expected - outHigh) ** 2 # Error function
                        row[i] = originalWeight - EPSILON
                        outLow = self.eval(features)[0]
                        errLow = 0.5 * (expected - outLow) ** 2
                        row[i] = originalWeight
                        rowDerivative.append((errHigh - errLow)/(2*EPSILON))
                    layerDerivative.append(rowDerivative)
                derivatives.append(layerDerivative)
            return derivatives

        ## Test to make sure backprop is correct by comparing proposed adjustments to computed derivatives.
        def gradientCheck(self, features, expected):
            derivatives = self.getDerivative(features, expected)
            actual = self.eval(features)[0]
            adjustments = list(self.adjustWeights([expected]))
            for ld, la in zip(derivatives, adjustments):
                for rd, ra in zip(ld, la):
                    for d, a in zip(rd, ra):
                        if d != 0 and a != 0:
                            err = abs(- a - d)
                            # There's a better way to do this that checks for numerical stability.
                            # But with ReLu we failed that check, so here we are.
                            # Turns out the naive approach doesn't work well when the derivatives are small due to floating point error.
                            if err > 0.000001:
                                print(err)
                                return False
            return True

        # Print all of the weights in the network
        def printWeights(self):
            for layer in self.layers:
                for row in layer.weights:
                    print(row)

    # Layer
    # Description: A single layer of a neural network
    #
    # Variables:
    #   size: The number of neurons.
    #   weights: A matrix of weights, set up for vector multipliaction of the 
    #       previous layer's activations to get this layer's activations. The
    #       y coordinate is the neuron in this layer, the x coordinate is the
    #       neuron in the previous layer. 
    #   learningRate: The learning rate parameter for adjusting weights.
    #   activation: The activation function
    #   activationPrime: The derivative of the activation function. Takes two
    #       arguments: input and activation output.
    #   inputs: The last evaluated inputs.
    #   sums: The last weighted sum of inputs (before the activation function)
    #   outputs: The last evaluated layer outputs/activations.
    #   deltas: The last computed error terms.
    class Layer:

        ## Creates a new Layer. See class description for explanation of inputs.
        def __init__(self, size, weights, learningRate, activation, activationPrime):
            self.inputs = None
            self.size = size
            self.weights = weights
            self.learningRate = learningRate
            self.activation = activation
            self.activationPrime = activationPrime

        ## Computes the outputs for each neuron given the list of sums.
        def activate(self, sums):
            return list(map(self.activation, sums))

        ## Computes the outputs for each neuron given the list of inputs
        def eval(self, inputs):
            self.inputs = inputs

            sums = []
            for i in range(self.size):
                sumWeights = self.weights[i]
                total = sumWeights[0] # Bias
                for j in range(1, len(sumWeights)):
                    total += inputs[j-1] * sumWeights[j]
                sums.append(total)

            self.sums = sums
            output = self.activate(sums)
            self.outputs = output
            return output

        ## Computes the error terms for each neuron given the error of each neuron.
        # WARNING: Must call eval first
        def getErrorTerms(self, error):
            self.deltas = []
            for i in range(self.size):
                delta = error[i] * self.activationPrime(self.sums[i], self.outputs[i]) #self.outputs[i] * (1 - self.outputs[i])
                self.deltas.append(delta)
            prevError = []
            for i in range(len(self.inputs)):
                err = 0
                for j in range(self.size):
                    err += self.weights[j][i+1] * self.deltas[j]
                prevError.append(err)

            return prevError

        ## Updates the weights following the perceptron adjustment rule.
        # WARNING: Must call getErrorTerms first.
        def adjustWeights(self, error):
            changes = []
            for i in range(len(self.weights)):
                changeRow = []
                for j in range(len(self.weights[i])):
                    weightInput = self.inputs[j-1] if j > 0 else 1 # Deal with bias
                    change = self.deltas[i] * weightInput
                    changeRow.append(change)
                    self.weights[i][j] += self.learningRate * change
                changes.append(changeRow)
            return changes

    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        self.isFirstTurn = True
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        # Setup
        if self.isFirstTurn:
            self.firstTurn(currentState)

        # Concede game if taking too long
        elif self.turnsPlayed > STALEMATE_TURNS:
            workers = getAntList(currentState, currentState.whoseTurn, (WORKER,))
            for worker in workers:
                if not worker.hasMoved:
                    return Move(MOVE_ANT, createPathToward(currentState, worker.coords, self.enemyHill.coords, UNIT_STATS[WORKER][MOVEMENT]))
            return Move(END)

        # Look ahead to each possible state and find the one with the lowest output
        bestH = 100000000
        bestMove = None
        moves = listAllLegalMoves(currentState)
        for move in moves:
            features = self.extractFeatures(getNextState(currentState, move))
            netH = self.network.eval(self.normalizeFeatures(features))[0]
            if netH < bestH:
                bestH = netH
                bestMove = move
            if ONLINE_LEARNING:
                h = self.heuristicStepsToGoal(features)
                self.network.adjustWeights((normalizeHeuristic(h),))
        if bestMove.moveType == END:
            self.turnsPlayed += 1
        return bestMove

        

    ## Finds the number of move actions it will take to reach a given destination
    def movesToReach(self, currentState, source, dest, unitType):
        taxicabDist = abs(dest[0] - source[0]) + abs(dest[1] - source[1])
        cost = float(taxicabDist)
        return cost

    ## Computes penalty for a worker standing on key structures.
    def getWorkerPenalty(self, currentState, workerCoords):
        if workerCoords == self.bestFoodConstr.coords:
            return TUNNEL_CONSTR_PENALTY
        elif workerCoords == self.bestFood.coords:
            return FOOD_CONSTR_PENALTY
        return 0

    ##
    # getWorkerCost
    # Params:
    #   currentState: game state
    #
    # Returns:
    #   the number of moves it will take for the worker to deliver a food plus penalties
    def getWorkerCost(self, currentState, workerCoords, carrying, isFakeAnt=False):
        cost = 0
        if carrying:
            cost = self.movesToReach(currentState, workerCoords, self.bestFoodConstr.coords,
                                     WORKER) + TUNNEL_CONSTR_PENALTY 
        else:
            cost = self.movesToReach(currentState, workerCoords, self.bestFood.coords,
                                     WORKER) + self.foodDist + FOOD_CONSTR_PENALTY + TUNNEL_CONSTR_PENALTY
        return cost

    ##
    # firstTurn
    # Description: inits variables. If TRAIN, will also train the Nerual Network
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #
    #
    def firstTurn(self, currentState):
        if TRAIN:
            self.train(TRAINING_FILE, self.network)
        
        inventory = getCurrPlayerInventory(currentState)
        tunnel = inventory.getTunnels()[0]
        hill = inventory.getAnthill()
        foods = getConstrList(currentState, None, (FOOD,))
        enemyInv = getEnemyInv(None, currentState)
        self.enemyHill = enemyInv.getAnthill()

        minDist = 100000  # arbitrarily large

        for food in foods:
            tunnelDist = self.movesToReach(currentState, tunnel.coords, food.coords, WORKER)
            hillDist = self.movesToReach(currentState, hill.coords, food.coords, WORKER)
            if tunnelDist < minDist:
                minDist = tunnelDist
                self.bestFood = food
                self.bestFoodConstr = tunnel
            if hillDist < minDist:
                minDist = hillDist
                self.bestFood = food
                self.bestFoodConstr = hill

        self.foodDist = minDist
        self.turnsPlayed = 0
        self.isFirstTurn = False
        self.totalSquareError = 0
        self.numMovesEvaluated = 0

    ##
    # heuristicStepsToGoal
    # Description: Calculates the number of steps required to get to the goal.
    # This function is originally from Ryan and Kollin's HW 2, but it is heavily modified.
    #
    # Parameters:
    #   features - A Features tuple
    #
    # Return:
    #   The estimated number of turns before the game is won.
    def heuristicStepsToGoal(self, features):
        foodLeft = FOOD_GOAL - features.foodCount + features.hasWorker # The food that must be gathered to win the game.
        adjustment = 0  # Penalty added for being in a board state likely to lose.

        # If the other player is ahead on food or we have a drone, send a drone to kill workers
        if features.enemyMoreFood and features.enemyHasWorker:
            if not features.hasDrone:
                adjustment += NEED_DRONE_PENALTY
                foodLeft += UNIT_STATS[DRONE][COST]

        adjustment += features.droneDist

        # If there are enemy units in our territory, fight them
        if features.areScaryFighters:
            if not features.hasSoldier:
                foodLeft += UNIT_STATS[SOLDIER][COST]
                adjustment += NEED_SOLDIER_PENALTY
            adjustment += features.numScaryFighters # Extra penalty

        adjustment += features.soldierDist

        # If we don't have a worker, we allot food to buy one
        if not features.hasWorker:
            foodLeft += UNIT_STATS[WORKER][COST]
            adjustment += NEED_WORKER_PENALTY

        # Could not get rid of worker jams without search
        # So this is an arbitrary penalty to punish the agent for building extra workers
        if features.mutlipleWorkers:
            adjustment += 20

        # Prevent queen from jamming workers
        adjustment += features.queenPenalty

        # After all workers deliver food, how many trips from the construct to the food and back will we need to end the game
        foodRuns = foodLeft - 1

        raw = features.workerCost # Raw estimate assuming we do not have an opponent

        # Now, calculate cost to complete all the necessary full trips to gather all food
        if foodRuns > 0:
            actions = features.roundTripCost * foodRuns
            # Doubled to account for end turn actions
            raw += 2 * actions

        # Max 1 food per turn, so we cannot go under the number of food remaining
        raw = max(raw, foodLeft)

        # Actual heuristic, accounting for cost from enemy winning
        # Casting to float makes the linter evaluate the return value correctly
        return float(raw + adjustment)

    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # Prints weights for online learning
    #
    def registerWin(self, hasWon):
        self.gamesPlayed += 1
        if ONLINE_LEARNING and self.gamesPlayed % 10 == 0:
            self.network.printWeights()
            print()
            print()

## These two functions (normalizeValue and normalizeHeuristic) scales a positive value to between 0 and 1.
#  Since the heuristic gets much larger, it scales much more slowly.
#  normalizeValue is meant to scale the inputs
def normalizeValue(x):
	return -math.exp(-x/50.0)+1

## Scales a positive value to between 0 and 1.
def normalizeHeuristic(x):
    return -math.exp(-x/200.0)+1

## Generates an x by y matrix with entries randomly between -1 and 1
def randomMatrix(x, y):
    return [[random.uniform(-1, 1) for _ in range(x)] for __ in range(y)]

## The sigmoid function
def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))

## The derivative of sigmoid (x is the input, y is the sigmoid output).
def sigmoidPrime(x, y):
    return y * (1 - y)

## Rectified Linear activation. Experimented with but not used for the output layer.
def relu(x):
    return x if x > 0 else 0.2*x

## The derivative of the relu functino.
def reluPrime(x, y):
    return 1 if x > 0 else 0.2