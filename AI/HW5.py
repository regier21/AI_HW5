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

MAX_DEPTH = 5
FOOD_CONSTR_PENALTY = MAX_DEPTH + 1
TUNNEL_CONSTR_PENALTY = 2 * FOOD_CONSTR_PENALTY


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

    def extractFeatures(self, currentState):
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

        winner = getWinner(currentState)
        food = inventory.foodCount
        enemyFoodCount = enemyInv.foodCount
        numDrones = len(drones)
        numSoldiers = len(soldiers)
        numEnemyWorkers = len(enemyWorkers)
        numScaryFighters = len(scaryFighters)
        numWorkers = len(workers)

        #distToFood = 0

        droneDist = 0
        if len(enemyWorkers) > 0:
            droneDist += sum(map(lambda enemyWorker: \
                                      self.movesToReach(currentState, droneSource, enemyWorker.coords, DRONE), enemyWorkers))
        droneDist += self.movesToReach(currentState, droneSource, enemyInv.getAnthill().coords, DRONE)

        soldierDist = 0
        soldierDist += sum(
            map(lambda target: self.movesToReach(currentState, soldierSource, target.coords, SOLDIER), scaryFighters))
        soldierDist += self.movesToReach(currentState, soldierSource, enemyInv.getAnthill().coords, SOLDIER)

        #numEndangeredUnits = 0
        queenPenalty = 120.0 / (approxDist(queen.coords, self.bestFoodConstr.coords) + 1) + 120.0 / (
                approxDist(queen.coords, self.bestFood.coords) + 1)

        workerCost = 0
        workerCoords = workers[0].coords if len(workers) > 0 else anthill.coords
        workerCarrying = workers[0].carrying if len(workers) > 0 else False
        workerCost += self.getWorkerPenalty(currentState, workerCoords)
        workerCost += self.getWorkerCost(currentState, workerCoords, workerCarrying)
        roundTripCost = self.getWorkerCost(currentState, anthill.coords, False)

        Features = namedtuple("Features", ["winner", "foodCount", "enemyFoodCount",
                "numDrones", "numSolders", "numWorkers", "numEnemyWorkers",
                "numScaryFighters", "droneDist", "soldierDist", "queenPenalty",
                "workerCost", "roundTripCost"])

        return Features(winner, food, enemyFoodCount, numDrones, numSoldiers, numWorkers, numEnemyWorkers,
                numScaryFighters, droneDist, soldierDist, queenPenalty, workerCost, roundTripCost)

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
        if self.isFirstTurn:
            self.firstTurn(currentState)
        return min(listAllLegalMoves(currentState), key=lambda x:
            self.heuristicStepsToGoal(self.extractFeatures(getNextState(currentState, x))))

        ##
        # firstTurn
        # Description: inits variables
        #
        # Parameters:
        #   currentState - A clone of the current state (GameState)
        #
        #

        ## Finds the number of move actions it will take to reach a given destination
    def movesToReach(self, currentState, source, dest, unitType):
        taxicabDist = abs(dest[0] - source[0]) + abs(dest[1] - source[1])
        cost = float(taxicabDist)
        # Ceiling for workers creates a set of equidistant points they may choose between
        # Since the move to stay still is always the last in the list,
        # this encourages the worker to move between equidistant points when stuck
        # This repositioning helps clear up most worker jams so they will always gather food
        # return cost if unitType != WORKER else float(math.ceil(cost))
        return cost
        ## Gets penalties for workers staying on a construct
        #  This penalty lets the ant go farther away from their target to leave the construct, helping to prevent jams

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
                                     WORKER) + TUNNEL_CONSTR_PENALTY  # Plus one from the penalty for standing on a tunnel
            # if not isFakeAnt:
            #     nextCoord = min(listAdjacent(workerCoords),
            #                     key=lambda dest: approxDist(dest, self.bestFoodConstr.coords))
            #     ant = getAntAt(currentState, nextCoord)
            #     if ant != None and not ant.carrying:
            #         cost += 3  # Lets the other worker move out and back to prevent jam
        else:
            cost = self.movesToReach(currentState, workerCoords, self.bestFood.coords,
                                     WORKER) + self.foodDist + FOOD_CONSTR_PENALTY + TUNNEL_CONSTR_PENALTY  # Plus two from the penalty for standing on the food and tunnel
        return cost

    def firstTurn(self, currentState):
        inventory = getCurrPlayerInventory(currentState)
        tunnel = inventory.getTunnels()[0]
        hill = inventory.getAnthill()
        foods = getConstrList(currentState, None, (FOOD,))
        enemyInv = getEnemyInv(None, currentState)
        enemyTunnel = enemyInv.getTunnels()[0]
        enemyHill = enemyInv.getAnthill()

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
        self.isFirstTurn = False
        self.bestAligned = False
        self.alignmentAxis = None

        if self.bestFood.coords[0] == self.bestFoodConstr.coords[0]:
            self.bestAligned = True
            self.alignmentAxis = 'x'
        elif self.bestFood.coords[1] == self.bestFoodConstr.coords[1]:
            self.bestAligned = True
            self.alignmentAxis = 'y'

        ##
        # heuristicStepsToGoal
        # Description: Calculates the number of steps required to get to the goal
        # Most of this function's code is to prevent a stalemate
        # A tiny amount of it actually wins the game
        #
        # Parameters:
        #   currentState - A clone of the current state (GameState)
        #
        #

    def heuristicStepsToGoal(self, features):
        # Get common variables
        foodLeft = FOOD_GOAL - features.foodCount + features.numWorkers

        winner = features.winner
        # Special case
        if winner == 1:
            return 0
        elif winner == 0:
            return 1000  # arbitraryily bad

        # Prevent a jam where we have no food or workers but keep killing Booger drones by having all units rush the anthill
        # if features.foodCount == 0 and features.numWorkers == 0:
        #     return sum(map(lambda ant: self.movesToReach(currentState, ant.coords, otherAnthillCoords, ant.type),
        #                    inventory.ants))

        # State variables used to compute total heuristic
        adjustment = 0  # Penalty added for being in a board state likely to lose.
        wantWorker = True  # Whether we should see a bonus from having extra workers.
        # Lets us buy defense instead of workers when necessary

        # If the other player is ahead on food or we have a drone, send a drone to kill workers
        if (features.enemyFoodCount >= features.foodCount and features.numEnemyWorkers > 0) or features.numDrones > 0:
            if features.numDrones == 0:
                adjustment += 1
                wantWorker = False
                foodLeft += UNIT_STATS[DRONE][COST]

            adjustment += features.droneDist

            # elif len(drones) > 0:
            #     # In this case, no reason to just have the drone lying around, so we charge the anthill
            #     # This cost needs to be negative so that the drone's cost does not go up by killing the last worker
            #     adjustment -= 1.0 / (
            #             self.movesToReach(currentState, drones[0].coords, otherInv.getAnthill().coords, DRONE) + 1)

        # If there are enemy units in our territory, fight them and retreat workers and queen
        if features.numScaryFighters > 0:
            # We are going to increment adjustment by the number of moves necessary for a soldier
            # to reach all the enemy units
            # We are also going to give us a food alloance to buy the soldier
            if features.numSolders == 0:
                wantWorker = False
                foodLeft += UNIT_STATS[SOLDIER][COST]
            adjustment += features.numScaryFighters

            # Retreat workers and queen
            # We ignore this once workers are dead b/c Booger stops playing so there is no longer reason to retreat
            # (and we will jam from perpetual retreat otherwise)
            # if features.numEnemyWorkers > 0:
            #     # Find squares under attack
            #     for enemy in enemyFighters:
            #         for coord in listAttackable(enemy.coords,
            #                                     UNIT_STATS[enemy.type][MOVEMENT] + UNIT_STATS[enemy.type][RANGE]):
            #             ant = getAntAt(currentState, coord)
            #             # Gently encourage retreat
            #             if ant != None and ant.player == me:
            #                 adjustment += 1 if ant.type == WORKER or ant.type == QUEEN else 0

            #             # If anthill in danger, double soldier food allowance and make threatening enemy high priority
            #             # Also, this prevents a jam where drone by anthill keeps killing worker while soldier
            #             #   is busy killing the newly-spawned drones
            #             # These penalties are arbitrary but seem to get the job done
            #             if coord == anthillCoords:
            #                 if len(soldiers) == 0:
            #                     wantWorker = False
            #                     foodLeft += UNIT_STATS[SOLDIER][COST]
            #                 adjustment += self.movesToReach(currentState, enemy.coords, start,
            #                                                 SOLDIER) * 10  # Arbitrary to make the priority

        adjustment += features.soldierDist

        # We need a fake worker count to prevent dividing by zero
        # If we don't have a worker, we also allot a food alloance to buy one if we don't have defense units we were saving for
        workerCount = features.numWorkers
        if workerCount == 0:
            foodLeft += UNIT_STATS[WORKER][COST]
            workerCount = 1

        # Could not get rid of worker jams without search
        # So this is an arbitrary penalty to punish the agent for building extra workers
        if workerCount > 1:
            adjustment += 20
            workerCount = 1

        # Prevent queen from jamming workers
        adjustment += features.queenPenalty

        # After all workers deliver food, how many trips from the construct to the food and back will we need to end the game
        foodRuns = foodLeft - features.numWorkers

        raw = features.workerCost # Raw estimate assuming we do not have an opponent
        if raw <= 1:
            return 0

        # Now, calculate cost to complete all the necessary full trips to gather all food
        if foodRuns > 0:
            actions = features.roundTripCost * foodRuns

            # Add actions plus estimated cost of end turns
            # To prevent incentivizing worker when we need defense, we prentend there is one worker for this calculation
            #   when we do not want a worker
            raw += 2 * actions

        # Max 1 food per turn, so we cannot go under the number of food remaining
        raw = max(raw, foodLeft)

        # Actual heuristic, accounting for cost from enemy winning
        # Casting to float makes the linter evaluate the return value correctly
        # (and makes our return value consistently float rather than occasionaly)
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
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        #method templaste, not implemented
        pass

## This function is 0 at 0 and approaches 1 as h approaches infinity
def normalizeHeuristic(h):
	return -math.exp1(-h/50.0) #50 was arbitrary to ensure 1 was approached slowly
