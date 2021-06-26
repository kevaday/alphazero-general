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
