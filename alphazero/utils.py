import alphazero.MCTS


class dotdict(dict):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        data = super().copy()
        return self.__class__(data)


def get_iter_file(iteration: int):
    return f'iteration-{iteration:04d}.pkl'


def scale_temp(scale_factor: float, min_temp: float, cur_temp: float, turns: int, const_max_turns: int) -> float:
    if (turns + 1) % int(scale_factor * const_max_turns) == 0:
        return max(min_temp, cur_temp / 2)
    else:
        return cur_temp


def default_temp_scaling(*args, **kwargs) -> float:
    return scale_temp(0.15, 0.2, *args, **kwargs)


def get_game_results(result_queue, game_cls, _get_index=None):
    player_to_index = {p: i for i, p in enumerate(range(game_cls.num_players()))}

    num_games = result_queue.qsize()
    wins = [0] * game_cls.num_players()
    draws = 0
    game_len_sum = 0

    for _ in range(num_games):
        state, winstate, agent_id = result_queue.get()
        game_len_sum += state.turns

        for player, is_win in enumerate(winstate):
            if is_win:
                if player == len(wins):
                    draws += 1
                else:
                    index = _get_index(player, agent_id) if _get_index else player_to_index[player]
                    wins[index] += 1

    return wins, draws, game_len_sum / num_games if num_games else 0


def plot_mcts_tree(mcts, max_depth=2):
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()

    global node_idx
    node_idx = 0

    def find_nodes(cur_node, _past_node=None, _past_i=None, _depth=0):
        if _depth > max_depth: return
        global node_idx
        cur_idx = node_idx

        G.add_node(cur_idx, a=cur_node.a, q=round(cur_node.q, 2), n=cur_node.n, v=round(cur_node.v, 2))
        if _past_node:
            G.add_edge(cur_idx, _past_i)
        node_idx += 1

        for node in cur_node._children:
            find_nodes(node, cur_node, cur_idx, _depth+1)

    find_nodes(mcts._root)
    labels = {node: '\n'.join(['{}: {}'.format(k, v) for k, v in G.nodes[node].items()]) for node in G.nodes}
    #pos = nx.spring_layout(G, k=0.15, iterations=50)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Gnodesep=1.0 -Goverlap=false')
    nx.draw(G, pos, labels=labels)
    plt.show()
