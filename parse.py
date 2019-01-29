import copy
import json
import os
import os.path
import zstd
from multiprocessing import Pool

from hlt import entity
from hlt import positionals
from hlt import game_map
from hlt import constants

ARBITRARY_ID = -1
name_dict = {2807: "teccles", 3191: "TheDuck314", 254: "cowzow", 257: "reCurs3", 1368: "SiestaGuru",
             247: "Team Schildpad",
             657: "adam", 231: "blasterpoard", 305: "cdurbin", 206: "mlomb", 134: "Rachol", 1023: "TonyK",
             1241: "FiodorG",
             2886: "zxqfl", 8414: "smurt"}


def parse_replay_file(file_name):
    with open(file_name, 'rb') as f:
        data = json.loads(zstd.loads(f.read()))
    constants.set_dimensions(data["production_map"]["width"], data["production_map"]["height"])
    player_id = int(file_name.split("-")[-1][:-4])
    player_name = name_dict[player_id] if player_id in name_dict else None
    player = [p for p in data['players'] if " ".join(p['name'].split(" ")[:-1]) == player_name]
    if len(player) > 0:
        player = player[0]
    else:
        print(f"Skipping {file_name}")
        return []
    player_id = int(player["player_id"])
    my_shipyard = entity.Shipyard(player_id, ARBITRARY_ID,
                                  positionals.Position(player['factory_location']['x'],
                                                       player['factory_location']['y']))
    other_shipyards = [
        entity.Shipyard(p['player_id'], ARBITRARY_ID,
                        positionals.Position(p['factory_location']['x'], p['factory_location']['y']))
        for p in data['players'] if int(p['player_id']) != player_id]
    width = data['production_map']['width']
    height = data['production_map']['height']
    first_cells = []
    for x in range(len(data['production_map']['grid'])):
        row = []
        for y in range(len(data['production_map']['grid'][x])):
            row += [game_map.MapCell(positionals.Position(x, y), data['production_map']['grid'][x][y]['energy'])]
        first_cells.append(row)
    frames = []
    for f in data['full_frames']:
        prev_cells = first_cells if len(frames) == 0 else frames[-1]._cells
        new_cells = copy.deepcopy(prev_cells)
        for c in f['cells']:
            new_cells[c['y']][c['x']].halite_amount = c['production']
        frames.append(game_map.GameMap(new_cells, width, height))

    moves = [{} if str(player_id) not in f['moves'] else {m['id']: m['direction'] for m in f['moves'][str(player_id)] if
                                                          m['type'] == "m"} for f in data['full_frames']]
    ships = [{} if str(player_id) not in f['entities'] else {
        int(sid): entity.Ship(player_id, int(sid), positionals.Position(ship['x'], ship['y']), ship['energy']) for
    sid, ship in
        f['entities'][str(player_id)].items()} for f in data['full_frames']]
    other_ships = [
        {int(sid): entity.Ship(int(pid), int(sid), positionals.Position(ship['x'], ship['y']), ship['energy']) for
         pid, p in
         f['entities'].items() if
         int(pid) != player_id for sid, ship in p.items()} for f in data['full_frames']]
    first_my_dropoffs = [my_shipyard]
    first_them_dropoffs = other_shipyards
    my_dropoffs = []
    them_dropoffs = []
    for f in data['full_frames']:
        new_my_dropoffs = copy.deepcopy(first_my_dropoffs if len(my_dropoffs) == 0 else my_dropoffs[-1])
        new_them_dropoffs = copy.deepcopy(first_them_dropoffs if len(them_dropoffs) == 0 else them_dropoffs[-1])
        for e in f['events']:
            if e['type'] == 'construct':
                if int(e['owner_id']) == player_id:
                    new_my_dropoffs.append(
                        entity.Dropoff(player_id, ARBITRARY_ID,
                                       positionals.Position(e['location']['x'], e['location']['y'])))
                else:
                    new_them_dropoffs.append(
                        entity.Dropoff(e['owner_id'], ARBITRARY_ID,
                                       positionals.Position(e['location']['x'], e['location']['y'])))
        my_dropoffs.append(new_my_dropoffs)
        them_dropoffs.append(new_them_dropoffs)
    return list(zip(frames, moves, ships, other_ships, my_dropoffs, them_dropoffs))


def parse_replay_folder(folder_name, max_files=None):
    print(f"Loading replays in {folder_name}.")
    replay_buffer = []
    ret = []
    for file_name in sorted(os.listdir(folder_name)):
        if not file_name.endswith(".hlt"):
            continue
        elif max_files is not None and len(replay_buffer) >= max_files:
            break
        else:
            replay_buffer.append(os.path.join(folder_name, file_name))
    with Pool(80) as p:
        buff = p.map(parse_replay_file, replay_buffer)
    for b in buff:
        if b:
            ret += b
    return ret
