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


def get_game_results(result_queue, game_cls, _get_index=None):
    player_to_index = {p: i for i, p in enumerate(game_cls.get_players())}

    num_games = result_queue.qsize()
    wins = [0] * len(game_cls.get_players())
    draws = 0
    for _ in range(num_games):
        _, value, agent_id = result_queue.get()
        if value != 0:
            index = _get_index(value, agent_id) if _get_index else player_to_index[value]
            wins[index] += 1
        else:
            draws += 1

    return wins, draws