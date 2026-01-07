import copy
import numpy as np
from rdkit.Chem import RemoveAllHs
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F

from confidence.dataset import ListDataset
from utils import so3, torus
from utils.molecules_utils import get_symmetry_rmsd
from utils.sampling import randomize_position, sampling
from utils.diffusion_utils import get_t_schedule


def loss_function(tr_pred, rot_pred, tor_pred, sidechain_pred, data, t_to_sigma, device, tr_weight=1, rot_weight=1,
                  tor_weight=1, backbone_weight=0, sidechain_weight=0, apply_mean=True, no_torsion=False,
                  nci_logits=None, nci_labels=None, nci_loss_weight=1.0):
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(
        *[torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type]
          for noise_type in ['tr', 'rot', 'tor']])
    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred.cpu() - tr_score.cpu()) ** 2 * tr_sigma.cpu() ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims).detach()

    # rotation component
    rot_score = torch.cat([d.rot_score for d in data], dim=0) if device.type == 'cuda' else data.rot_score
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1)
    rot_loss = (((rot_pred.cpu() - rot_score.cpu()) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score.cpu() / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion component
    if not no_torsion:
        edge_tor_sigma = torch.from_numpy(
            np.concatenate([d.tor_sigma_edge for d in data] if device.type == 'cuda' else data.tor_sigma_edge))
        tor_score = torch.cat([d.tor_score for d in data], dim=0) if device.type == 'cuda' else data.tor_score
        tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float()
        tor_loss = ((tor_pred.cpu() - tor_score.cpu()) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score.cpu() ** 2 / tor_score_norm2)).detach()
        if apply_mean:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float), tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones(d['ligand'].edge_mask.sum()) * i for i, d in
                               enumerate(data)]).long() if device.type == 'cuda' else data['ligand'].batch[
                data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(tor_loss.shape))
            c = c + 0.0001
            t_l.index_add_(0, index, tor_loss)
            t_b_l.index_add_(0, index, tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    if backbone_weight > 0:
        backbone_vecs = torch.cat([d['receptor'].side_chain_vecs.cpu() for d in data], dim=0) if device.type == 'cuda' else data['receptor'].side_chain_vecs
        backbone_vecs = backbone_vecs[:, 4:]
        backbone_pred = sidechain_pred[:, 4:]

        backbone_base_loss = (backbone_vecs ** 2).detach().mean(dim=1) + 0.0001
        backbone_loss = ((backbone_pred.cpu() - backbone_vecs) ** 2).mean(dim=1) / backbone_base_loss.mean()
        backbone_base_loss = backbone_base_loss / backbone_base_loss.mean()
        if apply_mean:
            backbone_loss, backbone_base_loss = backbone_loss.mean() * torch.ones(1, dtype=torch.float), backbone_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones((d['receptor'].pos.shape[0])) * i for i, d in enumerate(data)], dim=0).long() if device.type == 'cuda' else data['receptor'].batch
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(backbone_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, backbone_loss)
            s_b_l.index_add_(0, index, backbone_base_loss)
            backbone_loss, backbone_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            backbone_loss, backbone_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            backbone_loss, backbone_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    if sidechain_weight > 0:
        sidechain_vecs = torch.cat([d['receptor'].side_chain_vecs.cpu() for d in data],
                                  dim=0) if device.type == 'cuda' else data['receptor'].side_chain_vecs
        chi_angles = sidechain_vecs[:, :4].to(device)
        chi_pred = sidechain_pred[:, :4].to(device)

        chi_pred = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_pred)
        chi_angles = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_angles)

        difference = torch.abs(chi_pred - chi_angles)
        difference = torch.min(difference, 1 - difference) # angles are circular and 360 degrees = 1

        sidechain_base_loss = (chi_angles ** 2).detach().mean(dim=1) + 0.0001
        sidechain_loss = (difference ** 2).mean(dim=1) / sidechain_base_loss.mean()
        sidechain_base_loss = sidechain_base_loss / sidechain_base_loss.mean()
        if apply_mean:
            sidechain_loss, sidechain_base_loss = \
                sidechain_loss.mean().cpu() * torch.ones(1, dtype=torch.float), \
                sidechain_base_loss.mean().cpu() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones((d['receptor'].pos.shape[0])) * i for i, d in enumerate(data)],
                              dim=0).long() if device.type == 'cuda' else data['receptor'].batch
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(sidechain_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, sidechain_loss.cpu())
            s_b_l.index_add_(0, index, sidechain_base_loss.cpu())
            sidechain_loss, sidechain_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            sidechain_loss, sidechain_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            sidechain_loss, sidechain_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(
                len(rot_loss), dtype=torch.float)

    nci_loss, pos_recall, none_acc = _nci_loss_and_metrics(
        nci_logits=nci_logits,
        nci_labels=nci_labels,
        data=data,
        device=device,
        apply_mean=apply_mean
    )
    nci_loss = nci_loss.cpu()
    pos_recall = pos_recall.cpu()
    none_acc = none_acc.cpu()

    loss_diff = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight + \
        sidechain_loss * sidechain_weight + backbone_loss * backbone_weight
    loss = loss_diff + nci_loss_weight * nci_loss
    return loss, tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), backbone_loss.detach(), sidechain_loss.detach(), \
           nci_loss.detach(), pos_recall.detach(), none_acc.detach(), tr_base_loss, rot_base_loss, tor_base_loss, \
           backbone_base_loss, sidechain_base_loss


def _nci_loss_and_metrics(nci_logits, nci_labels, data, device, apply_mean, alpha_none=0.25, alpha_pos=1.0, gamma=2.0):
    if nci_logits is None:
        return _empty_nci_metrics(data, device, apply_mean)

    labels = _resolve_nci_labels(data, nci_labels, device)
    if labels is None:
        return _empty_nci_metrics(data, device, apply_mean)

    if device.type == 'cuda':
        logits_per_graph = _split_nci_logits(nci_logits, data)
        labels_per_graph = _split_nci_labels(labels, data)
        return _compute_nci_per_graph(logits_per_graph, labels_per_graph, device, apply_mean,
                                      alpha_none=alpha_none, alpha_pos=alpha_pos, gamma=gamma)

    edge_batch = data['ligand'].batch[data['ligand', 'nci_cand', 'receptor'].edge_index[0]]
    num_graphs = data.num_graphs
    return _compute_nci_batched(nci_logits, labels, edge_batch, num_graphs, device, apply_mean,
                                alpha_none=alpha_none, alpha_pos=alpha_pos, gamma=gamma)


def _resolve_nci_labels(data, nci_labels, device):
    if nci_labels is not None:
        return nci_labels
    if device.type == 'cuda':
        labels = []
        for d in data:
            if ('ligand', 'nci_cand', 'receptor') in d.edge_types and \
                    hasattr(d['ligand', 'nci_cand', 'receptor'], 'edge_type_y'):
                labels.append(d['ligand', 'nci_cand', 'receptor'].edge_type_y)
            else:
                labels.append(None)
        if all(l is None for l in labels):
            return None
        return labels
    if ('ligand', 'nci_cand', 'receptor') not in data.edge_types:
        return None
    if not hasattr(data['ligand', 'nci_cand', 'receptor'], 'edge_type_y'):
        return None
    return data['ligand', 'nci_cand', 'receptor'].edge_type_y


def _split_nci_logits(nci_logits, data):
    edge_counts = []
    for d in data:
        if ('ligand', 'nci_cand', 'receptor') in d.edge_types:
            edge_counts.append(d['ligand', 'nci_cand', 'receptor'].edge_index.size(1))
        else:
            edge_counts.append(0)
    if nci_logits.numel() == 0:
        return [nci_logits] * len(edge_counts)
    return list(torch.split(nci_logits, edge_counts, dim=0))


def _split_nci_labels(labels, data):
    if isinstance(labels, (list, tuple)):
        return list(labels)
    edge_counts = []
    for d in data:
        if ('ligand', 'nci_cand', 'receptor') in d.edge_types:
            edge_counts.append(d['ligand', 'nci_cand', 'receptor'].edge_index.size(1))
        else:
            edge_counts.append(0)
    if labels.numel() == 0:
        return [labels] * len(edge_counts)
    return list(torch.split(labels, edge_counts, dim=0))


def _compute_nci_per_graph(logits_per_graph, labels_per_graph, device, apply_mean, alpha_none, alpha_pos, gamma):
    num_graphs = len(logits_per_graph)
    loss_out = torch.zeros(num_graphs, dtype=torch.float, device=device)
    pos_recall = torch.zeros(num_graphs, dtype=torch.float, device=device)
    none_acc = torch.zeros(num_graphs, dtype=torch.float, device=device)
    valid_mask = torch.zeros(num_graphs, dtype=torch.bool, device=device)
    for i, (logits, labels) in enumerate(zip(logits_per_graph, labels_per_graph)):
        if labels is None or logits is None or labels.numel() == 0:
            continue
        loss_i, pos_recall_i, none_acc_i = _compute_nci_single(logits, labels, alpha_none, alpha_pos, gamma)
        loss_out[i] = loss_i
        pos_recall[i] = pos_recall_i
        none_acc[i] = none_acc_i
        valid_mask[i] = True
    if apply_mean:
        if not valid_mask.any():
            zeros = torch.zeros(1, dtype=torch.float, device=device)
            return zeros, zeros, zeros
        return loss_out[valid_mask].mean() * torch.ones(1, dtype=torch.float, device=device), \
            pos_recall[valid_mask].mean() * torch.ones(1, dtype=torch.float, device=device), \
            none_acc[valid_mask].mean() * torch.ones(1, dtype=torch.float, device=device)
    return loss_out, pos_recall, none_acc


def _compute_nci_batched(nci_logits, labels, edge_batch, num_graphs, device, apply_mean, alpha_none, alpha_pos, gamma):
    if labels is None or nci_logits is None or labels.numel() == 0:
        return _empty_nci_metrics(num_graphs, device, apply_mean)
    focal_loss = _focal_loss(nci_logits, labels, alpha_none, alpha_pos, gamma)
    loss_sum = torch.zeros(num_graphs, dtype=torch.float, device=device)
    counts = torch.zeros(num_graphs, dtype=torch.float, device=device)
    loss_sum.index_add_(0, edge_batch, focal_loss)
    counts.index_add_(0, edge_batch, torch.ones_like(focal_loss))
    loss_per_graph = loss_sum / (counts + 1e-6)

    preds = torch.argmax(nci_logits, dim=1)
    pos_mask = labels > 0
    none_mask = labels == 0
    pos_correct = ((preds == labels) & pos_mask).float()
    none_correct = ((preds == 0) & none_mask).float()
    pos_total = torch.zeros(num_graphs, dtype=torch.float, device=device)
    none_total = torch.zeros(num_graphs, dtype=torch.float, device=device)
    pos_correct_sum = torch.zeros(num_graphs, dtype=torch.float, device=device)
    none_correct_sum = torch.zeros(num_graphs, dtype=torch.float, device=device)
    pos_total.index_add_(0, edge_batch, pos_mask.float())
    none_total.index_add_(0, edge_batch, none_mask.float())
    pos_correct_sum.index_add_(0, edge_batch, pos_correct)
    none_correct_sum.index_add_(0, edge_batch, none_correct)
    pos_recall = pos_correct_sum / (pos_total + 1e-6)
    none_acc = none_correct_sum / (none_total + 1e-6)

    if apply_mean:
        return loss_per_graph.mean() * torch.ones(1, dtype=torch.float, device=device), \
            pos_recall.mean() * torch.ones(1, dtype=torch.float, device=device), \
            none_acc.mean() * torch.ones(1, dtype=torch.float, device=device)
    return loss_per_graph, pos_recall, none_acc


def _compute_nci_single(logits, labels, alpha_none, alpha_pos, gamma):
    focal_loss = _focal_loss(logits, labels, alpha_none, alpha_pos, gamma)
    loss = focal_loss.mean()
    preds = torch.argmax(logits, dim=1)
    pos_mask = labels > 0
    none_mask = labels == 0
    pos_correct = ((preds == labels) & pos_mask).float().sum()
    none_correct = ((preds == 0) & none_mask).float().sum()
    pos_recall = pos_correct / (pos_mask.float().sum() + 1e-6)
    none_acc = none_correct / (none_mask.float().sum() + 1e-6)
    return loss, pos_recall, none_acc


def _focal_loss(logits, labels, alpha_none, alpha_pos, gamma):
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    labels = labels.long()
    log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    alpha = torch.where(labels == 0,
                        torch.full_like(labels, alpha_none, dtype=logits.dtype),
                        torch.full_like(labels, alpha_pos, dtype=logits.dtype))
    focal_term = (1 - pt) ** gamma
    return -alpha * focal_term * log_pt


def _empty_nci_metrics(data_or_num_graphs, device, apply_mean):
    if isinstance(data_or_num_graphs, int):
        num_graphs = data_or_num_graphs
    elif device.type == 'cuda':
        num_graphs = len(data_or_num_graphs)
    else:
        num_graphs = data_or_num_graphs.num_graphs
    if apply_mean:
        zeros = torch.zeros(1, dtype=torch.float, device=device)
        return zeros, zeros, zeros
    zeros = torch.zeros(num_graphs, dtype=torch.float, device=device)
    return zeros, zeros, zeros


def _unpack_model_outputs(outputs):
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 5:
            return outputs
        if len(outputs) == 4:
            tr_pred, rot_pred, tor_pred, sidechain_pred = outputs
            return tr_pred, rot_pred, tor_pred, sidechain_pred, None
    raise ValueError("Unexpected model output format; expected 4 or 5 outputs.")


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum().cpu() if self.unpooled_metrics else v.cpu()
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weights):
    model.train()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
                          'nci_loss', 'pos_recall', 'none_acc', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss',
                          'backbone_base_loss', 'sidechain_base_loss'])

    for data in tqdm(loader, total=len(loader)):
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
            continue
        optimizer.zero_grad()
        data = [d.to(device) for d in data] if device.type == 'cuda' else data
        try:
            tr_pred, rot_pred, tor_pred, sidechain_pred, nci_logits = _unpack_model_outputs(model(data))
            loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma,
                                 device=device, nci_logits=nci_logits)
            if loss_tuple is None:
                print("None loss tuple, skipping")
                continue
            loss = loss_tuple[0]

            if torch.any(torch.isnan(loss)):
                names = data.name if device.type == 'cpu' else [d.name for d in data]
                print("Nan loss, skipping batch with complexes", names)
                continue
            loss.backward()
            optimizer.step()
            if ema_weights is not None: ema_weights.update(model.parameters())
            meter.add([loss.cpu().detach(), *loss_tuple[1:]])
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                #raise e
                print(e)
                continue
            
    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
                          'nci_loss', 'pos_recall', 'none_acc', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss',
                          'backbone_base_loss', 'sidechain_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
             'nci_loss', 'pos_recall', 'none_acc', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss',
             'backbone_base_loss', 'sidechain_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                tr_pred, rot_pred, tor_pred, sidechain_pred, nci_logits = _unpack_model_outputs(model(data))
            loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma,
                                 apply_mean=False, device=device, nci_logits=nci_logits)
            if loss_tuple is None: continue
            meter.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]])

            if test_sigma_intervals > 0:
                complex_t_tr, complex_t_rot, complex_t_tor = [torch.cat([data[i].complex_t[noise_type] for i in range(len(data))]) for
                                                              noise_type in ['tr', 'rot', 'tor']]
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                meter_all.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]],
                    [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr,
                     sigma_index_tr, sigma_index_tr, sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor,
                     sigma_index_tr, sigma_index_tr])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e
                print(e)
                continue

    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out


def inference_epoch_fix(model, complex_graphs, device, t_to_sigma, args):
    t_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=args.inference_steps,
                                inf_sched_alpha=1, inf_sched_beta=1)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds, min_rmsds = [], []

    for orig_complex_graph in tqdm(loader):
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(args.inference_samples)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)

        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(data_list=data_list, model=model.module if device.type == 'cuda' else model,
                                                         inference_steps=args.inference_steps,
                                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                         tor_schedule=tor_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args,
                                                         t_schedule=t_schedule)
            except Exception as e:
                failed_convergence_counter += 1
                if failed_convergence_counter > 5:
                    print('failed 5 times - skipping the complex')
                    break
                print("Exception while running inference on complex:", e)
        if failed_convergence_counter > 5:
            rmsds.extend([100] * args.inference_samples)
            min_rmsds.append(100)
            continue

        if args.no_torsion:
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph[
                                                         'ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
        # if len(orig_complex_graph['ligand'].orig_pos.shape) == 3:
        #     orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        if len(orig_complex_graph['ligand'].orig_pos.shape) == 2:
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[None, :, :]
        try:
            orig_ligand_pos = orig_complex_graph['ligand'].orig_pos[:, filterHs] - orig_complex_graph.original_center.cpu().numpy()
        except Exception as e:
            print("problem with orig_pos which is of shape:", orig_complex_graph['ligand'].orig_pos.shape, e)
            continue
        mol = RemoveAllHs(orig_complex_graph.mol[0])
        complex_rmsds = []
        for i in range(len(orig_ligand_pos)):
            try:
                rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[i], [l for l in ligand_pos])
            except Exception as e:
                print("Using non corrected RMSD because of the error:", e)
                rmsd = np.sqrt(((ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=2).mean(axis=1))
            complex_rmsds.append(rmsd)
        complex_rmsds = np.asarray(complex_rmsds)
        rmsd = np.min(complex_rmsds, axis=0)
        
        rmsds.extend([r for r in rmsd])
        min_rmsds.append(rmsd.min(axis=0))

    rmsds = np.array(rmsds)
    min_rmsds = np.array(min_rmsds)
    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds)),
              'min_rmsds_lt2': (100 * (min_rmsds < 2).sum() / len(min_rmsds)),
              'min_rmsds_lt5': (100 * (min_rmsds < 5).sum() / len(min_rmsds)),}
    return losses
