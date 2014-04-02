# coding: utf-8
"""
zombsole-paladin 3.0 - AI for zombsole

Copyright (c) 2014 Gustavo Picón - https://tabo.pe/

Paladin is an AI program for zombsole. It's based on an A*
algorithm to find paths depending on a strategy, and
includes tactics and endgame routines.

The pathing considers the following heuristics:

  - Paladins will quickly form groups to increase their chance of
    survival. They will start forming small groups of 2-3 bots,
    and then these groups will form larger groups until all the
    paladins are together.
  - Paladins will try to travel next to walls and boxes to avoid
    being surrounded.
  - Paladins avoid zombies when traveling, and if given no
    choice, they prefer paths with a lower concentration
    of zombies.
  - Paladins will prefer routes that have nearby Paladins, even
    if they are slightly longer.
  - Paladins will break through walls and boxes if that
    can provide a faster/safer route to a destination.
  - When alone, paladins will tend to panic and flee when zombies
    get too close. This should keep strangled paladins alive until
    a group can get to them.

The design is based on a supervisor mode, in which it is assumed
that the Paladins in a match can communicate to each other in
order to better battle zombies. This is used to cluster in groups,
travel in formation, avoiding targetting the same enemy
unnecesarily, and entering special modes. If a Paladin finds other
AIs, it will still colaborate by joining bands and healing other
bots, but their effectiveness will be lower.

Paladins can play on any valid map for the `evacuation`,
`extermination`, and `safehouse` rules. Some special
end-game considerations:

  - In `safehouse` mode, when Paladins get near the safehouse, they
    will enter `GET TO THE CHOPPA!` mode, meaning they will abandon
    formation and will rush to the safehouse ASAP.
  - In `extermination` mode, when there are roughly 2 zombies per
    Paladin, the bots will enter `ZERG RUSH` mode, which means they
    will abandon formation and smite the closest zombie, with a
    holy shotgun.
  - In `evacuation` mode, once paladins formed a single group and
    are in healing range of each other, they will enter
    `ALL TOGETHER NOW` mode, by disabling the A* pathing algorithm
    and breaking through walls and boxes to form a single impenetrable
    unit. This is Sparta.

Also, in safehouse and evacuation mode, Paladins will prefer
courses of action that avoid fights, since killing zombies
won't win games.


Cheap optimizations:

A* can be quite expensive given the heuristics, the size of the map, and
python itself. A compromise is done, that takes advantages of groups.
Every time a group is formed, the paladin closer to an objective is
chosen as a group leader. This leader will then calculate a path to the
objective, and the other group members will calculate paths to the place
where the leader will be in 2 turns. Thus, the less and larger the groups,
the faster the game will be.

Another optimization is that, when grouping, clustering movement is
calculated by distance, not by optimal path distance
(see `player_distances()`). This was the original approach (that
had better results) but it was too slow to be "fun". Better results
are compensated by including tactics and endgame routines.


"Cheating"

Zombsole maps are infinite, but only a small part of the map is visible.
Zombies tend to spawn in the visible part or very close to it, so
the non-visible areas are safer. The heuristics in `evacuation` and
`safehouse` modes avoid zombies, so in maps with many zombies, paladins
will tend to gather in non-visible areas of the map. This is something
that I did not intend, but is a nice effect of the heuristics working
within the rules of the game.


There are no tests, comments or docstrings.
Abandon all hope, ye who enter here.

"""

import math
from functools import cmp_to_key
import itertools
from collections import defaultdict

from core import HEALING_RANGE
from things import Player, Zombie, Wall, Box
from utils import (
    closest, distance, adjacent_positions, sort_by_distance, to_position,
    possible_moves
)
from weapons import Shotgun


INFINITE = float('inf')

TIGHT_GROUP_MAX_DISTANCE = math.sqrt(HEALING_RANGE**2 + 1**2)
NORMAL_GROUP_MAX_DISTANCE = HEALING_RANGE * 2


class AStar(object):

    def __init__(self, movement_cost, heuristics, neighbor_nodes):
        self.heuristics = heuristics
        self.neighbor_nodes = neighbor_nodes
        self.movement_cost = movement_cost

    def process(self, start, goal):
        closed_nodes = set()
        open_nodes = {start}
        parents = {}
        g_scores = {start: 0}
        f_scores = {start: self.heuristics(start, goal)}
        key_cmp_fscores = cmp_to_key(lambda x, y: f_scores[x] - f_scores[y])
        while open_nodes:
            current = min(open_nodes, key=key_cmp_fscores)
            if current == goal:
                node = current
                ret = [node]
                while node in parents:
                    node = parents[node]
                    ret.insert(0, node)
                return ret
            open_nodes.remove(current)
            closed_nodes.add(current)
            for neighbor in self.neighbor_nodes(current):
                cost = g_scores[current] + self.movement_cost(current, neighbor)
                if neighbor in open_nodes and cost < g_scores[neighbor]:
                    # found a better path
                    open_nodes.remove(neighbor)
                if neighbor in closed_nodes and cost < g_scores[neighbor]:
                    # non-monotonic heuristics
                    closed_nodes.remove(neighbor)
                if neighbor not in open_nodes and neighbor not in closed_nodes:
                    g_scores[neighbor] = cost
                    open_nodes.add(neighbor)
                    f_scores[neighbor] = cost + self.heuristics(neighbor, goal)
                    parents[neighbor] = current
        return []


class Strategy(object):

    MIN_HEALTH_BEFORE_URGENT_SELF_HEAL = 40
    MIN_HEALTH_BEFORE_SELF_HEAL = 80
    MIN_HEALTH_BEFORE_FRIEND_HEAL = 50

    PATH_THROUGH_WALL_RATIO = 8.0
    PATH_THROUGH_BOX_RATIO = 0.5
    PATH_NEAR_WALL_RATIO = -0.1
    PATH_NEAR_BOX_RATIO = -0.05
    PATH_NEAR_PLAYER_RATIO = -0.2
    ZOMBIE_AVERSION_RATIO = 0.2

    should_flee = False

    def __init__(self):
        self.things = {}
        self.map = {}
        self.turn = None
        self.objectives = []
        self.paths = {}
        self.rules = None
        self.forced_goals = {}
        self.targets = {}
        self.positions = defaultdict(list)
        self._a_star_obj = AStar(
            distance, self.cost_heuristics, adjacent_positions)
        self.tactics = {}

    def cost_heuristics(self, initial, goal):
        cost = distance(initial, goal)
        obj = self.map[goal]
        perc = 1.0
        if isinstance(obj, Wall):
            perc += self.PATH_THROUGH_WALL_RATIO
        elif isinstance(obj, Box):
            perc += self.PATH_THROUGH_BOX_RATIO
        for pos in adjacent_positions(goal):
            neighobj = self.map[pos]
            if isinstance(neighobj, Wall):
                perc += self.PATH_NEAR_WALL_RATIO
            elif isinstance(neighobj, Box):
                perc += self.PATH_NEAR_BOX_RATIO
            elif isinstance(neighobj, Player):
                perc += self.PATH_NEAR_PLAYER_RATIO
        zombie_threats = [
            zombie for zombie in self.map.zombies
            if distance(zombie, goal) <= zombie.weapon.max_range
        ]
        perc += self.ZOMBIE_AVERSION_RATIO * len(zombie_threats)
        if perc <= 0:
            perc = 0.01
        return cost * perc

    def plan_strategy(self):
        raise NotImplementedError

    def process_turn(self, turn, things, rules, objectives):
        if self.turn != turn:
            self.things = things
            self.map = Map(things, objectives)
            self.rules = rules
            self.objectives = objectives
            self.turn = turn
            self.targets = {}
            self.paths = self.plan_strategy()

    def find_path(self, start, goal):
        goal_is_player = self.map.is_player(goal)
        goal_pos = None
        if goal_is_player:
            goal_pos = closest(start, self.map.possible_moves(goal))
        if goal_pos is None:
            goal_pos = goal
        path = self._a_star_obj.process(start, goal_pos)
        if start in path:
            path.remove(start)
        if goal in path and goal_is_player:
            path.remove(goal)
        return path

    def all_together_now(self, players):
        respaths = {}
        avgpos = average_position(players)
        for player in players:
            current_distance = distance(player, avgpos)
            for newpos in adjacent_positions(player):
                if (
                    not self.map.is_player(newpos) and
                    distance(newpos, avgpos) < current_distance
                ):
                    respaths[player] = [newpos]
        return respaths

    def group_travel(self, players, destination_set):
        respaths = {}
        player_set = set(players)
        leader = find_closest_in_group_to_targets(player_set, destination_set)
        close = closest(leader, destination_set)
        if close:
            leader_path = self.find_path(leader, to_position(close))
            if leader_path:
                respaths[leader] = leader_path
                flock_to_pos = leader_path[:3].pop()
                for player in player_set - {leader}:
                    path = self.find_path(player.position, flock_to_pos)
                    if path:
                        respaths[player] = path
        return respaths

    def come_together(self):
        if len(self.map.players) <= 1:
            return {}
        distances = player_distances(self.map.players)
        if max(distances.values()) <= HEALING_RANGE * 2:
            return self.all_together_now(
                self.map.players)
        player_set = set(self.map.players)
        groups = self.map.find_player_groups(NORMAL_GROUP_MAX_DISTANCE)
        return {
            player: path
            for group in groups
            for player, path in self.group_travel(
                group, player_set - group).items()
        }

    def action_for(self, player):
        if player not in self.tactics:
            self.tactics[player] = Tactics(self, player)
        return self.tactics[player].process(self.map)


class SafeHouseStrategy(Strategy):

    should_flee = True

    ZOMBIE_AVERSION_RATIO = 2.0

    def plan_strategy(self):
        if self.can_run_to_the_choppa():
            return self.get_to_the_choppa()
        if not self.map.are_players_clustered():
            return self.come_together()
        return self.group_travel(self.map.players, self.objectives)

    def players_not_in_safehouse(self):
        objs = self.objectives
        return sorted(
            [p for p in self.map.players if p.position not in objs],
            key=lambda x: d_to_things(x, objs)
        )

    def can_run_to_the_choppa(self):
        for player in self.players_not_in_safehouse():
            for pos in adjacent_positions(player):
                if pos in self.objectives:
                    return True
        return False

    def get_to_the_choppa(self):
        return {
            player: self.find_path(
                player.position,
                sort_by_distance(
                    self.players_not_in_safehouse()[0],
                    self.objectives
                )[-1]
            )
            for player in self.map.players
        }


class EvacuationStrategy(Strategy):

    should_flee = True

    ZOMBIE_AVERSION_RATIO = 0.5

    def plan_strategy(self):
        return self.come_together()


class ExterminationStrategy(Strategy):

    should_flee = False

    ZOMBIE_AVERSION_RATIO = -0.2

    def plan_strategy(self):
        if self.can_rush():
            return self.zerg_rush()
        return self.come_together()

    def can_rush(self):
        return (
            self.map.are_players_clustered() or
            (1.0 * len(self.map.zombies) / len(self.map.players)) < 2
        )

    def zerg_rush(self):
        targets = {}
        for player in self.map.players:
            closest_zombie = closest(player, self.map.zombies)
            path = self.find_path(player.position, closest_zombie.position)
            if path:
                targets[player] = path
        return targets


class Tactics(object):
    def __init__(self, strategy, player):
        self.strategy = strategy
        self.player = player
        self.map = None
        self.plan = ''

    def tentative_next_pos(self):
        self.strategy.positions[self.player].append(self.player.position)
        path = self.strategy.paths.get(self.player, [])
        self.plan = self.strategy_plan_msg()
        next_pos = None
        while path and path[0] == self.player.position:
            path.pop(0)
        while path and self.map.is_player(path[-1]):
            path.pop()
        if path:
            next_pos = path[0]
        return next_pos

    def opportunistic_attack(self):
        zombie_targets = self.map.targets(self.player)
        min_dmg = self.player.weapon.damage_range[0]
        while zombie_targets:
            zombie_target = zombie_targets.pop(0)
            already_targetted = [
                p for p, player_target in self.strategy.targets.items()
                if zombie_target == player_target
            ]
            if len(already_targetted) * min_dmg > zombie_target.life:
                continue
            self.strategy.targets[self.player] = zombie_target
            self.status("Attack zombie{} (H={}).".format(zombie_target.position,
                                                         zombie_target.life))
            return 'attack', zombie_target

    def avoid_endless_loop(self, next_pos):
        poslog = self.strategy.positions[self.player][:]
        if len(poslog) >= 4 and next_pos is not None:
            current_pos = poslog.pop()
            player_position = self.player.position
            assert current_pos == player_position
            prev_pos = poslog.pop()
            if poslog.pop() == player_position and poslog.pop() == prev_pos:
                for pos in adjacent_positions(self.player):
                    zombie = closest(pos, self.map.zombies)
                    if (
                        next_pos and
                        pos != next_pos and
                        self.map[pos] is None and
                        (
                            zombie is None or
                            distance(pos, zombie) > zombie.weapon.max_range
                        )
                    ):
                        return pos
        return next_pos

    def process(self, map_object):
        self.map = map_object
        player = self.player
        pos = player.position
        next_pos = self.tentative_next_pos()

        try:
            friend = self.map.healable(pos).pop(0)
        except IndexError:
            friend = None

        move_quickly_to = self.move_to_safety(next_pos)
        if move_quickly_to:
            going_next = self.goto_next_pos(move_quickly_to)
            if going_next:
                return going_next

        if player.life < self.strategy.MIN_HEALTH_BEFORE_SELF_HEAL:
            self.status('Heal self (urgent!).')
            return 'heal', player

        if friend and friend.life < self.strategy.MIN_HEALTH_BEFORE_FRIEND_HEAL:
            self.status('Heal *INJURED* player{} (H={}).'.format(
                friend.position, friend.life))
            return 'heal', friend

        next_pos = self.avoid_endless_loop(next_pos)
        if next_pos and self.pain(next_pos) < self.pain(pos):
            going_next = self.goto_next_pos(next_pos)
            if going_next:
                return going_next

        attack = self.opportunistic_attack()
        if attack:
            return attack

        going_next = self.goto_next_pos(next_pos)
        if going_next:
            return going_next

        if player.life < 100:
            self.status('Heal self.'.format(pos))
            return 'heal', player
        if friend and friend.life < 100:
            self.status('Heal player{}.'.format(friend.position))
            return 'heal', friend

        self.status("Praying to Mystra...")
        return None

    def status(self, msg):
        self.player.status = "{} P={}".format(msg, self.plan)

    def strategy_plan_msg(self):
        path = self.strategy.paths.get(self.player, [])
        if not path:
            return 'Wait.'
        pos = path[-1]
        numticks = len(path)
        msg = '{} move{} to '.format(numticks, '' if numticks == 1 else 's')
        if self.map.is_zombie(pos):
            return msg + 'kill Zombie{}.'.format(pos)
        return msg + '{}.'.format(pos)

    def goto_next_pos(self, next_pos):
        if not next_pos:
            return
        thing_in_nextpos = self.map[next_pos]
        if thing_in_nextpos is None:
            pos = self.player.position
            self.status('Move {}->{}.'.format(pos, next_pos))
            return 'move', next_pos
        if not self.map.is_player(next_pos):
            self.status('Attack {}{} (H={}).'.format(
                thing_in_nextpos.name, next_pos, thing_in_nextpos.life))
            return 'attack', thing_in_nextpos

    def is_fleeing_off_screen(self):
        players = set(self.map.players)
        players.remove(self.player)
        if d_to_things(self.player, players) <= HEALING_RANGE:
            return False
        x, y = self.player.position
        sizex, sizey = self.map.size
        toofar = HEALING_RANGE * 4
        return min(x, y) < -toofar or x > sizex + toofar or y > sizey + toofar

    def is_safe_pos(self, pos):
        return pos and self.map.can_move_to(pos) and not self.pain(pos)

    def is_player_in_danger(self):
        threats = len(self.map.threats(self.player.position))
        players = self.map.players[:]
        players.remove(self.player)
        if d_to_things(self.player, players) <= HEALING_RANGE:
            return False
        if d_to_things(self.player, players) <= HEALING_RANGE * 1.5:
            max_pain = self.player.life - 15
            max_threats = 5
            should_flee_on_threats = False
        else:
            max_pain = self.player.life - 30
            max_threats = 3
            should_flee_on_threats = self.strategy.should_flee
        return (
            self.pain(self.player.position) > max_pain or
            threats >= max_threats or
            (threats and should_flee_on_threats)
        )

    def move_to_safety(self, next_pos):
        if not self.is_player_in_danger():
            return
        if self.is_safe_pos(next_pos):
            self.plan = 'Hurrying up to next position for safety.'
            return next_pos
        if self.is_fleeing_off_screen():
            self.plan = 'Paladin tired of running away! Standing ground.'
            return
        fleeing_options = self.map.possible_moves(self.player)
        if fleeing_options:
            flee_to = min(fleeing_options, key=self.pain)
            new_pain = self.pain(flee_to)
            if not new_pain or new_pain < self.pain(self.player.position):
                self.plan = '**FLEEING!**'
                return flee_to

    def pain(self, pos):
        return sum(
            zombie.weapon.damage_range[1] for zombie in self.map.threats(pos))


class Map(object):
    def __init__(self, things, objectives):
        self.things = things
        self.objectives = objectives
        self._cache = {}
        self._zombies = None

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self.things, key)
        except KeyError:
            return None

    def find_living_things(self, cls):
        if cls not in self._cache:
            self._cache[cls] = [
                thing for thing in self.things.values() if
                isinstance(thing, cls) and thing.life > 0
            ]
        return self._cache[cls]

    @property
    def zombies(self):
        return self.find_living_things(Zombie)

    @property
    def players(self):
        return self.find_living_things(Player)

    def threats(self, pos):
        return sort_by_distance(
            pos,
            [
                zombie for zombie in self.zombies
                if distance(zombie, pos) <= zombie.weapon.max_range
            ]
        )

    def healable(self, pos):
        key_cmp_low_health = cmp_to_key(lambda x, y: x.life - y.life)
        return sorted(
            [
                player for player in self.players
                if (
                    player.position != pos and
                    distance(player, pos) <= HEALING_RANGE
                )
            ],
            key=key_cmp_low_health
        )

    def targets(self, player):
        return sort_by_distance(
            player,
            [
                zombie for zombie in self.zombies
                if distance(player, zombie) <= player.weapon.max_range
            ]
        )

    def are_players_clustered(self):
        players = {player for player in self.players if
                   player.position not in self.objectives}
        if len(players) <= 1:
            return True
        for player in players:
            players_copy = players.copy()
            players_copy.remove(player)
            players_copy = sort_by_distance(player, list(players_copy))
            if distance(player, players_copy[0]) > HEALING_RANGE:
                return False
            if distance(player, players_copy[-1]) > NORMAL_GROUP_MAX_DISTANCE:
                return False
        return True

    def find_player_groups(self, max_distance):
        players_in_groups = set()
        groups_found = []
        for comblength in range(len(self.players), 0, -1):
            for group in itertools.combinations(self.players, comblength):
                group = set(group)
                if players_in_groups & group:
                    continue
                if len(group) > 1:
                    distances = player_distances(group)
                    max_in_group = max(dst for dst in distances.values())
                else:
                    max_in_group = 0
                if max_in_group <= max_distance:
                    groups_found.append(group)
                    players_in_groups |= group
        return groups_found

    def possible_moves(self, player):
        return possible_moves(player, self.things)

    def can_move_to(self, pos):
        return self.things.get(pos) is None

    @property
    def size(self):
        keys = self.things.keys()
        return max(x[0] for x in keys), max(y[1] for y in keys)

    def is_player(self, pos):
        return isinstance(self.things.get(pos), Player)

    def is_zombie(self, pos):
        return isinstance(self.things.get(pos), Zombie)


def find_closest_in_group_to_targets(candidates, targets):
    return min(candidates, key=lambda x: distance(x, closest(x, targets)))


def player_distances(players):
    distances = {}
    for p1, p2 in itertools.combinations(players, 2):
        distances[(p1, p2)] = distances[(p2, p1)] = distance(p1, p2)
    return distances


def average_position(things):
    x, y = [], []
    l = len(things) * 1.0
    for thing in things:
        pos = to_position(thing)
        x.append(pos[0])
        y.append(pos[1])
    return round(sum(x) / l, 0), round(sum(y) / l, 0)


def d_to_things(thing, things):
    return distance(thing, closest(thing, things))


class Paladin(Player):

    _paladin_strategy = None

    @property
    def paladin_strategy(self):
        cls = self.__class__
        if cls._paladin_strategy is None:
            cls._paladin_strategy = {
                'safehouse': SafeHouseStrategy,
                'extermination': ExterminationStrategy,
                'evacuation': EvacuationStrategy
            }[self.rules]()
        return cls._paladin_strategy

    def next_step(self, things, turn):
        self.paladin_strategy.process_turn(
            turn, things, self.rules, self.objectives)
        return self.paladin_strategy.action_for(self)


def create(rules, objectives=None):
    return Paladin(
        'paladin',
        'red',
        weapon=Shotgun(),
        rules=rules,
        objectives=objectives
    )
