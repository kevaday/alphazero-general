from alphazero.NNetArchitecture import NNetArchitecture
from alphazero.NeuralNet import NeuralNet
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from time import time

import torch.optim as optim
import numpy as np
import torch
import pickle
import os


class NNetWrapper(NeuralNet):
    def __init__(self, game_cls, args):
        self.nnet = NNetArchitecture(game_cls, args)
        self.action_size = game_cls.action_size()
        self.optimizer = args.optimizer(self.nnet.parameters(), lr=args.lr, **args.optimizer_args)

        self.scheduler = args.scheduler(self.optimizer, **args.scheduler_args)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **args.scheduler_args)
        self.verbose = args.scheduler_args.get('verbose')

        if args.cuda:
            self.nnet.cuda()

        self.args = args

    def train(self, batches, train_steps):
        self.nnet.train()

        data_time = AverageMeter()
        batch_time = AverageMeter()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()

        if self.verbose:
            print(f'Current LR: {self.optimizer.param_groups[0]["lr"]}')

        bar = Bar(f'Training Net', max=train_steps)
        current_step = 0
        while current_step < train_steps:
            for batch_idx, batch in enumerate(batches):
                if current_step == train_steps:
                    break

                start = time()
                current_step += 1
                boards, target_pis, target_vs = batch

                # predict
                if self.args.cuda:
                    boards, target_pis, target_vs = (
                        boards.contiguous().cuda(),
                        target_pis.contiguous().cuda(),
                        target_vs.contiguous().cuda()
                    )

                # measure data loading time
                data_time.update(time() - start)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time() - start)

                # plot progress
                bar.suffix = '({step}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                    step=current_step,
                    size=train_steps,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                bar.next()

        self.scheduler.step(pi_losses.avg + v_losses.avg)
        bar.finish()
        print()

        return pi_losses.avg, v_losses.avg

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if self.args.cuda:
            board = board.contiguous().cuda()
        with torch.no_grad():
            self.nnet.eval()
            pi, v = self.nnet(board)

            # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
            return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def process(self, batch: torch.Tensor):
        batch = batch.type(torch.FloatTensor)
        if self.args.cuda:
            batch = batch.cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(batch)
            return torch.exp(pi), v

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save({
            'state_dict': self.nnet.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'sch_state': self.scheduler.state_dict()
        }, filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise IOError("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        if 'opt_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['opt_state'])
        if 'sch_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['sch_state'])
