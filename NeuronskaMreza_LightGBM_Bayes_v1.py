#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
    50% višeglavna neuronska mreža
    35% LightGBM sa vremenskim ponderisanjem
    15% uslovna empirijska Bayesova distribucija

Svaki CSV obrađuje se nezavisno.

Predikcija:
    Sedam glava opisuje distribucije sortiranih pozicija kombinacije.
    To nisu fizički redosledi izvlačenja.

Neuronska mreža:
    Residual MLP + LayerNorm + sedam regresionih glava.
    Kvadratni gubitak za kumulativne distribucije.
    Deterministička analitička inicijalizacija.
    Bez dropout-a, shuffle-a i slučajnog generisanja.

LightGBM:
    Regresija kumulativnih distribucija.
    Eksponencijalno vremensko ponderisanje.
    Bez slučajnog uzorkovanja redova i osobina.

Bayes:
    Dirichletova regularizacija na istorijskim celim kombinacijama.
    Uslovljavanje sličnošću prethodnih konteksta.
    Ne koristi se rang najčešćih pojedinačnih brojeva.

Izbor kombinacije:
    Zbir tri ponderisana zajednička skora distribucija.
    Deterministička pretraga sa grananjem i ograničavanjem.
    Poštuje samo sedam različitih brojeva iz 1..39.
    Nema naknadnog prepravljanja kombinacije.

Provera:
    48 kola za izbor dužine treniranja.
    Poslednja 64 kola za odvojenu završnu proveru.
    Finalni modeli treniraju se na svim raspoloživim prelazima.
"""


import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit(
        "Nedostaje biblioteka. Instalacija:\n"
        "python3 -m pip install numpy torch lightgbm\n"
        f"Detalj: {exc}"
    )


DEFAULT_FILES = (
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_2974.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_plus_1730.csv",
)

LAGS = (1, 2, 4, 8, 16, 32)
BURN = max(LAGS)

VALIDATION = 48
TEST = 64
MIN_ROWS = 300

MAX_EPOCHS = 40
BATCH_SIZE = 256
MAX_BOOST_ROUNDS = 60
DECAY_RATE = 0.0015

ENSEMBLE_WEIGHTS = np.array([0.50, 0.35, 0.15], dtype=float)
GRID = np.arange(1, 40)
EPS = 1e-12


def read_csv(filename):
    path = Path(filename).expanduser().resolve(strict=True)
    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or all(not value.strip() for value in row):
                continue

            if len(row) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    f"očekivano 7 kolona, pronađeno {len(row)}."
                )

            try:
                numbers = [int(value.strip()) for value in row]
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "potrebni su celi brojevi."
                ) from exc

            if len(set(numbers)) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: ponovljen broj."
                )

            if min(numbers) < 1 or max(numbers) > 39:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "broj izvan opsega 1..39."
                )

            rows.append(sorted(numbers))

    if len(rows) < MIN_ROWS:
        raise ValueError(
            f"{path.name}: potrebno najmanje {MIN_ROWS} redova."
        )

    draws = np.asarray(rows, dtype=np.int64)

    canonical = "\n".join(
        ",".join(map(str, draw)) for draw in draws
    ).encode("ascii")

    fingerprint = hashlib.sha256(canonical).hexdigest()
    return path, draws, fingerprint


def describe_draw(draw):
    numbers = draw.astype(float)
    distances = np.abs(numbers[:, None] - numbers[None, :])
    distances = distances[np.triu_indices(7, 1)]

    return np.r_[
        numbers / 39.0,
        [np.mean(distances <= threshold)
         for threshold in (2, 5, 10, 20)],
        (numbers[-1] - numbers[0]) / 38.0,
    ]


def make_data(draws):
    profiles = np.asarray([describe_draw(row) for row in draws])
    contexts = []

    for t in range(BURN, len(draws) + 1):
        history = [profiles[t - lag] for lag in LAGS]

        contexts.append(np.concatenate(
            history + [
                history[0] - history[1],
                history[1] - history[2],
                history[2] - history[3],
            ]
        ))

    x = np.asarray(contexts, dtype=np.float32)
    targets = draws[BURN:]

    # Posmatrana kumulativna funkcija za svaku sortiranu poziciju.
    cdf = (
        targets[:, :, None] <= GRID[None, None, :]
    ).astype(np.float32)

    return x, targets, cdf


def scaler_fit(x):
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def scale_data(x, mean, scale):
    return ((x - mean) / scale).astype(np.float32)


def isotonic(values):
    """Najbliži neopadajući niz, algoritmom PAVA."""
    means = []
    counts = []

    for value in np.clip(values, 0.0, 1.0):
        means.append(float(value))
        counts.append(1)

        while len(means) >= 2 and means[-2] > means[-1]:
            size = counts[-2] + counts[-1]
            average = (
                means[-2] * counts[-2]
                + means[-1] * counts[-1]
            ) / size

            means[-2:] = [average]
            counts[-2:] = [size]

    return np.repeat(means, counts)


def cdf_to_pmf(cdf):
    """
    Projekcija na matematički validnu kumulativnu funkciju.
    Ne nameće obrazac samoj kombinaciji.
    """
    cdf = np.asarray(cdf, dtype=float)

    if cdf.shape != (7, 39) or not np.all(np.isfinite(cdf)):
        raise ValueError("Nevažeći regresioni izlaz distribucije.")

    result = np.empty_like(cdf)

    for position in range(7):
        corrected = isotonic(cdf[position])
        corrected[-1] = 1.0

        mass = np.diff(np.r_[0.0, corrected])
        mass = np.maximum(mass, 0.0)
        total = mass.sum()

        if not math.isfinite(total) or total <= 0:
            raise ValueError("Distribucija nema pozitivnu masu.")

        result[position] = mass / total

    return result


class DeterministicLinear(nn.Module):
    """Linearni sloj bez pozivanja generatora slučajnih brojeva."""

    def __init__(self, inputs, outputs, phase):
        super().__init__()

        index = torch.arange(
            outputs * inputs, dtype=torch.float64
        ).reshape(outputs, inputs)

        values = (
            torch.sin((index + 1.0) * (0.713 + phase))
            + torch.cos((index + 1.0) * (0.371 + 0.5 * phase))
        )

        values *= math.sqrt(1.0 / inputs)

        self.weight = nn.Parameter(values.to(torch.float32))
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class LotteryDeepNet(nn.Module):
    def __init__(self, inputs):
        super().__init__()

        self.fc1 = DeterministicLinear(inputs, 128, 0.11)
        self.ln1 = nn.LayerNorm(128)

        self.fc2 = DeterministicLinear(128, 128, 0.23)
        self.ln2 = nn.LayerNorm(128)

        self.fc3 = DeterministicLinear(128, 64, 0.37)
        self.ln3 = nn.LayerNorm(64)

        self.heads = nn.ModuleList([
            DeterministicLinear(64, 39, 0.51 + position * 0.13)
            for position in range(7)
        ])

    def forward(self, x):
        h1 = F.silu(self.ln1(self.fc1(x)))
        h2 = F.silu(self.ln2(self.fc2(h1))) + h1
        h3 = F.silu(self.ln3(self.fc3(h2)))

        # Regresija CDF-a, bez softmax-a i klasifikacionog gubitka.
        return torch.stack(
            [torch.sigmoid(head(h3)) for head in self.heads],
            dim=1,
        )


class DeepBranch:
    def fit(self, x, y, epochs, validation=None):
        self.mean, self.scale = scaler_fit(x)
        z = torch.from_numpy(scale_data(x, self.mean, self.scale))
        target = torch.from_numpy(np.asarray(y, dtype=np.float32))

        self.model = LotteryDeepNet(x.shape[1])
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.002,
            weight_decay=1e-4,
        )

        if validation is not None:
            vx, vy = validation
            vz = torch.from_numpy(
                scale_data(vx, self.mean, self.scale)
            )
            vt = torch.from_numpy(np.asarray(vy, dtype=np.float32))

        best_loss = float("inf")
        best_state = None
        best_epoch = epochs
        stale = 0

        for epoch in range(1, epochs + 1):
            self.model.train()

            # Uzorci ostaju u hronološkom redosledu.
            for start in range(0, len(z), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(z))
                predicted = self.model(z[start:end])

                mse = F.mse_loss(predicted, target[start:end])

                # Učenje neopadanja CDF-a.
                monotonic = F.relu(
                    predicted[:, :, :-1] - predicted[:, :, 1:]
                ).square().mean()

                loss = mse + 0.10 * monotonic

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if validation is not None:
                self.model.eval()

                with torch.no_grad():
                    current_loss = float(
                        F.mse_loss(self.model(vz), vt).item()
                    )

                if current_loss < best_loss - 1e-8:
                    best_loss = current_loss
                    best_epoch = epoch
                    stale = 0
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.model.state_dict().items()
                    }
                else:
                    stale += 1

                if stale >= 8:
                    break

            if epoch % 10 == 0:
                print(f"    DeepNet epoha {epoch}/{epochs}", flush=True)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.epochs = best_epoch
        return self

    def predict(self, x):
        z = torch.from_numpy(
            scale_data(x, self.mean, self.scale)
        )

        self.model.eval()

        with torch.no_grad():
            return self.model(z).cpu().numpy()


class LightGBMBranch:
    def fit(self, x, y, rounds=None, validation=None):
        self.models = []
        self.rounds = np.ones((7, 39), dtype=np.int64)

        ages = np.arange(len(x) - 1, -1, -1, dtype=float)
        weights = np.exp(np.maximum(-DECAY_RATE * ages, -600.0))

        parameters = {
            "objective": "regression_l2",
            "metric": "l2",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 24,
            "lambda_l2": 1.0,
            "feature_fraction": 1.0,
            "feature_fraction_bynode": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "extra_trees": False,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 1,
            "bin_construct_sample_cnt": max(len(x), 1),
            "verbosity": -1,
        }

        for position in range(7):
            head = []

            for threshold in range(39):
                target = y[:, position, threshold]

                if np.ptp(target) == 0:
                    head.append(float(target[0]))
                    continue

                training = lgb.Dataset(
                    x,
                    label=target,
                    weight=weights,
                    free_raw_data=True,
                )

                number_of_rounds = (
                    MAX_BOOST_ROUNDS
                    if rounds is None
                    else max(1, int(rounds[position, threshold]))
                )

                callbacks = [lgb.log_evaluation(period=0)]
                valid_sets = None

                if validation is not None:
                    vx, vy = validation

                    valid_sets = [
                        lgb.Dataset(
                            vx,
                            label=vy[:, position, threshold],
                            reference=training,
                            free_raw_data=True,
                        )
                    ]

                    callbacks.append(
                        lgb.early_stopping(
                            stopping_rounds=8,
                            first_metric_only=True,
                            verbose=False,
                        )
                    )

                model = lgb.train(
                    parameters,
                    training,
                    num_boost_round=number_of_rounds,
                    valid_sets=valid_sets,
                    callbacks=callbacks,
                )

                best = model.best_iteration or number_of_rounds
                self.rounds[position, threshold] = best
                head.append(model)

            self.models.append(head)

            print(
                f"    LightGBM glava {position + 1}/7",
                flush=True,
            )

        return self

    def predict(self, x):
        output = np.empty((len(x), 7, 39), dtype=float)

        for position, head in enumerate(self.models):
            for threshold, model in enumerate(head):
                if isinstance(model, float):
                    output[:, position, threshold] = model
                else:
                    output[:, position, threshold] = model.predict(
                        x,
                        num_iteration=int(
                            self.rounds[position, threshold]
                        ),
                        num_threads=1,
                    )

        return output


class EmpiricalBayesBranch:
    """
    Distribucija nad istorijskim celim kombinacijama.

    Empirijski prior daje jednaku težinu istorijskim opažanjima,
    što nije uniformna raspodela brojeva 1..39.

    Kontekstualne težine daju frakcione pseudo-brojeve opažanja.
    Dirichletova posteriorna sredina računa se analitički,
    bez uzorkovanja.

    Ovo je uslovna, regularizovana adaptacija izvorne Bayesove
    grane; nije rang ukupnih frekvencija pojedinačnih brojeva.
    """

    def fit(self, x, y, alpha=10.0):
        self.mean, self.scale = scaler_fit(x)
        self.z = scale_data(x, self.mean, self.scale).astype(float)
        self.targets = y.astype(float)
        self.alpha = float(alpha)
        return self

    def predict_one(self, x):
        query = scale_data(
            np.asarray(x)[None, :], self.mean, self.scale
        )[0].astype(float)

        distances = np.mean((self.z - query) ** 2, axis=1)
        positive = distances[distances > EPS]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0

        similarity = (
            1.0 + distances / max(bandwidth, EPS)
        ) ** -4.0

        # Svaki istorijski prelaz zadržava pozitivnu težinu.
        conditional_counts = (
            len(self.z) * similarity / similarity.sum()
        )

        prior = np.full(len(self.z), self.alpha / len(self.z))
        posterior = prior + conditional_counts
        posterior /= posterior.sum()

        return np.tensordot(
            posterior, self.targets, axes=(0, 0)
        )


def component_score(probabilities, combination):
    values = np.asarray(combination, dtype=int)
    return np.prod(
        probabilities[:, np.arange(7), values],
        axis=1,
    )


def choose_combination(component_pmfs):
    """
    Maksimizuje:
        0.50 * proizvod DeepNet masa
      + 0.35 * proizvod LightGBM masa
      + 0.15 * proizvod Bayes masa

    Pretraga koristi gornje granice za svaku preostalu granu.
    Ne postavlja ograničenje na broj posećenih čvorova.
    """
    component_pmfs = np.asarray(component_pmfs, dtype=float)

    if component_pmfs.shape != (3, 7, 39):
        raise ValueError("Neispravan oblik distribucija ansambla.")

    if not np.all(np.isfinite(component_pmfs)):
        raise ValueError("Ansambl sadrži nevažeće vrednosti.")

    probabilities = np.zeros((3, 7, 40), dtype=float)
    probabilities[:, :, 1:] = component_pmfs

    # suffix[c, j, last]:
    # najveći mogući proizvod preostalih masa komponente c.
    suffix = np.zeros((3, 8, 40), dtype=float)
    suffix[:, 7, :] = 1.0

    for position in range(6, -1, -1):
        maximum_number = 33 + position

        for last in range(38, -1, -1):
            number = last + 1

            candidate = (
                probabilities[:, position, number]
                * suffix[:, position + 1, number]
                if number <= maximum_number
                else np.zeros(3)
            )

            suffix[:, position, last] = np.maximum(
                suffix[:, position, last + 1],
                candidate,
            )

    initial_candidates = []

    # Tačan optimum svake pojedinačne komponente
    # daje početnu donju granicu za zajednički skor.
    for component in range(3):
        selected = []
        last = 0

        for position in range(7):
            candidates = np.arange(last + 1, 34 + position)

            values = (
                probabilities[component, position, candidates]
                * suffix[component, position + 1, candidates]
            )

            number = int(candidates[np.argmax(values)])
            selected.append(number)
            last = number

        initial_candidates.append(tuple(selected))

    best_combination = None
    best_score = -1.0

    for combination in sorted(set(initial_candidates)):
        score = float(
            ENSEMBLE_WEIGHTS
            @ component_score(probabilities, combination)
        )

        if score > best_score:
            best_score = score
            best_combination = combination

    if best_score <= 0:
        raise ValueError(
            "Ansambl nema pozitivnu masu ni za jednu "
            "početnu legalnu kombinaciju."
        )

    visited = 0

    def visit(prefix, masses):
        nonlocal best_score, best_combination, visited
        visited += 1

        position = len(prefix)

        if position == 7:
            score = float(ENSEMBLE_WEIGHTS @ masses)

            if (
                score > best_score
                or (
                    score == best_score
                    and prefix < best_combination
                )
            ):
                best_score = score
                best_combination = prefix

            return

        last = prefix[-1] if prefix else 0

        upper = float(
            ENSEMBLE_WEIGHTS
            @ (masses * suffix[:, position, last])
        )

        if upper <= best_score:
            return

        branches = []

        for number in range(last + 1, 34 + position):
            next_masses = (
                masses * probabilities[:, position, number]
            )

            bound = float(
                ENSEMBLE_WEIGHTS
                @ (
                    next_masses
                    * suffix[:, position + 1, number]
                )
            )

            if bound > best_score:
                branches.append((bound, number, next_masses))

        branches.sort(key=lambda item: (-item[0], item[1]))

        for bound, number, next_masses in branches:
            if bound <= best_score:
                continue

            visit(prefix + (number,), next_masses)

    visit((), np.ones(3))

    return best_combination, best_score, visited


def longest_run(combination):
    maximum = current = 1

    for previous, number in zip(combination[:-1], combination[1:]):
        current = current + 1 if number == previous + 1 else 1
        maximum = max(maximum, current)

    return maximum


def fit_branches(x, y, epochs, rounds):
    deep = DeepBranch().fit(x, y, epochs=epochs)
    trees = LightGBMBranch().fit(x, y, rounds=rounds)
    bayes = EmpiricalBayesBranch().fit(x, y)
    return deep, trees, bayes


def predict_components(deep, trees, bayes, x):
    deep_cdf = deep.predict(x)
    tree_cdf = trees.predict(x)

    result = []

    for index in range(len(x)):
        bayes_cdf = bayes.predict_one(x[index])

        result.append(np.asarray([
            cdf_to_pmf(deep_cdf[index]),
            cdf_to_pmf(tree_cdf[index]),
            cdf_to_pmf(bayes_cdf),
        ]))

    return result


def process_dataset(path, draws, fingerprint):
    x, actual_targets, y = make_data(draws)

    count = len(y)
    test_start = count - TEST
    validation_start = test_start - VALIDATION

    print("\n" + "=" * 76)
    print(path.name)
    print(f"Izvlačenja: {len(draws)}")
    print(f"Istorijskih prelaza za finalno treniranje: {count}")
    print(f"SHA-256: {fingerprint}")
    print(
        f"Validacija: CSV redovi "
        f"{BURN + validation_start + 1}..{BURN + test_start}"
    )
    print(
        f"Završna provera: CSV redovi "
        f"{BURN + test_start + 1}..{len(draws)}"
    )

    training_x = x[:validation_start]
    training_y = y[:validation_start]

    validation_x = x[validation_start:test_start]
    validation_y = y[validation_start:test_start]

    print("\n1. Izbor dužine treniranja na validaciji", flush=True)

    selection_deep = DeepBranch().fit(
        training_x,
        training_y,
        epochs=MAX_EPOCHS,
        validation=(validation_x, validation_y),
    )

    selected_epochs = selection_deep.epochs
    del selection_deep

    selection_trees = LightGBMBranch().fit(
        training_x,
        training_y,
        validation=(validation_x, validation_y),
    )

    selected_rounds = selection_trees.rounds.copy()
    del selection_trees

    print(f"  Izabrano DeepNet epoha: {selected_epochs}")
    print(
        "  LightGBM iteracije po regresoru: "
        f"{selected_rounds.min()}..{selected_rounds.max()}"
    )

    print("\n2. Treniranje pre odvojene završne provere", flush=True)

    deep, trees, bayes = fit_branches(
        x[:test_start],
        y[:test_start],
        selected_epochs,
        selected_rounds,
    )

    test_components = predict_components(
        deep, trees, bayes,
        x[test_start:count],
    )

    predictions = []
    hit_counts = []
    control_hits = []
    score_variation = []

    for index, components in enumerate(test_components):
        combination, _, nodes = choose_combination(components)
        target_index = test_start + index

        actual = actual_targets[target_index]
        previous = draws[BURN + target_index - 1]

        predictions.append(combination)
        hit_counts.append(
            len(set(combination).intersection(actual))
        )
        control_hits.append(
            len(set(previous).intersection(actual))
        )
        score_variation.append(float(np.ptp(components)))

        if (index + 1) % 16 == 0:
            print(
                f"  Provera {index + 1}/{TEST}; "
                f"čvorova poslednje pretrage: {nodes}",
                flush=True,
            )

    hit_counts = np.asarray(hit_counts)
    control_hits = np.asarray(control_hits)

    print("\nREZULTATI ODVOJENE PROVERE")
    print(f"Prosečno pogodaka: {hit_counts.mean():.4f}")
    print(
        "Raspodela pogodaka 0..7: "
        f"{np.bincount(hit_counts, minlength=8).tolist()}"
    )
    print(f"Kola sa 3+: {np.sum(hit_counts >= 3)}")
    print(f"Kola sa 4+: {np.sum(hit_counts >= 4)}")
    print(
        "Kontrola: prethodno izvuceno kolo — "
        f"{control_hits.mean():.4f}"
    )
    print(
        "Razlika prema kontroli: "
        f"{hit_counts.mean() - control_hits.mean():+.4f}"
    )
    print(
        f"Različite prognoze: {len(set(predictions))}/{TEST}"
    )
    print(
        "Prognoze sa 5+ uzastopnih brojeva: "
        f"{sum(longest_run(item) >= 5 for item in predictions)}"
    )
    print(
        "Prognoze potpuno u jednoj polovini: "
        f"{sum(max(item) <= 19 or min(item) >= 20 for item in predictions)}"
    )

    del deep, trees, bayes, test_components

    print("\n3. Finalno treniranje na celoj istoriji", flush=True)

    deep, trees, bayes = fit_branches(
        x[:-1],
        y,
        selected_epochs,
        selected_rounds,
    )

    components = predict_components(
        deep, trees, bayes, x[-1:]
    )[0]

    combination, score, nodes = choose_combination(components)

    print("\nFINALNI NEXT: " + ", ".join(map(str, combination)))
    print(f"Zajednički modelski skor: {score:.12g}")
    print(f"Posećenih čvorova pretrage: {nodes}")
    print(f"Najduži uzastopni niz: {longest_run(combination)}")

    lower = sum(number <= 19 for number in combination)
    print(f"Brojeva 1..19 / 20..39: {lower} / {7 - lower}")
    print(
        "Ponderi: DeepNet=0.50, LightGBM=0.35, Bayes=0.15"
    )
    print(
        "Modelski skor je nenormalizovan zajednički skor "
        "za rangiranje legalnih kombinacija."
    )

    return combination


def main():
    parser = argparse.ArgumentParser(
        description="LotteryPrediction adaptacija za tri Loto 7/39 CSV-a."
    )
    parser.add_argument("csv", nargs="*")
    args = parser.parse_args()

    filenames = args.csv or DEFAULT_FILES

    if len(filenames) != 3:
        raise ValueError("Navedi tačno tri CSV putanje.")

    datasets = [read_csv(filename) for filename in filenames]

    if len({path for path, _, _ in datasets}) != 3:
        raise ValueError("Potrebna su tri različita CSV fajla.")

    # Računanje je namerno na CPU-u.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    print("LOTTERYPREDICTION LOTO V1")
    print("Tri komponente; izvorni ponderi 50/35/15.")
    print("Tri nezavisna CSV-a; cela istorija za finalni NEXT.")
    print("Bez random generisanja i klasifikacionog gubitka.")
    print("CSV redovi moraju biti od najstarijeg ka najnovijem.")
    print(
        "Treniranje više regresora i zajednička pretraga "
        "mogu potrajati; napredak se ispisuje."
    )

    results = []

    for path, draws, fingerprint in datasets:
        combination = process_dataset(path, draws, fingerprint)
        results.append((path.name, combination))

    print("\n" + "=" * 76)
    print("ZAVRŠNI PREGLED")

    for name, combination in results:
        print(f"{name}: " + ", ".join(map(str, combination)))

    grouped = {}

    for name, combination in results:
        grouped.setdefault(combination, []).append(name)

    for combination, names in grouped.items():
        if len(names) > 1:
            print(
                "\nIsti NEXT: "
                + " / ".join(names)
                + ": "
                + ", ".join(map(str, combination))
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrekinuto.", file=sys.stderr)
        sys.exit(130)
    except (
        OSError,
        ValueError,
        RuntimeError,
        csv.Error,
        lgb.basic.LightGBMError,
    ) as exc:
        print(f"\nGREŠKA: {exc}", file=sys.stderr)
        sys.exit(1)



"""
LOTTERYPREDICTION LOTO V1
Tri komponente; izvorni ponderi 50/35/15.
Tri nezavisna CSV-a; cela istorija za finalni NEXT.
Bez random generisanja i klasifikacionog gubitka.
CSV redovi moraju biti od najstarijeg ka najnovijem.
Treniranje više regresora i zajednička pretraga mogu potrajati; napredak se ispisuje.


============================================================================
loto7_4704_k83.csv
Izvlačenja: 4704
Istorijskih prelaza za finalno treniranje: 4672
SHA-256: 1decd748cc0c398e384d9b16fd7e7eb50a5396f59a2f2cf9e7cdd05bdd07ae09
Validacija: CSV redovi 4593..4640
Završna provera: CSV redovi 4641..4704

1. Izbor dužine treniranja na validaciji
    DeepNet epoha 10/40
    DeepNet epoha 20/40
    LightGBM glava 1/7
    LightGBM glava 2/7
    LightGBM glava 3/7
    LightGBM glava 4/7
    LightGBM glava 5/7
    LightGBM glava 6/7
    LightGBM glava 7/7
  Izabrano DeepNet epoha: 13
  LightGBM iteracije po regresoru: 1..60

2. Treniranje pre odvojene završne provere
    DeepNet epoha 10/13
    LightGBM glava 1/7
    LightGBM glava 2/7
    LightGBM glava 3/7
    LightGBM glava 4/7
    LightGBM glava 5/7
    LightGBM glava 6/7
    LightGBM glava 7/7
  Provera 16/64; čvorova poslednje pretrage: 24
  Provera 32/64; čvorova poslednje pretrage: 71
  Provera 48/64; čvorova poslednje pretrage: 32
  Provera 64/64; čvorova poslednje pretrage: 14

REZULTATI ODVOJENE PROVERE
Prosečno pogodaka: 1.2344
Raspodela pogodaka 0..7: [15, 26, 16, 7, 0, 0, 0, 0]
Kola sa 3+: 7
Kola sa 4+: 0
Kontrola: prethodno izvuceno kolo — 1.1406
Razlika prema kontroli: +0.0938
Različite prognoze: 63/64
Prognoze sa 5+ uzastopnih brojeva: 0
Prognoze potpuno u jednoj polovini: 0

3. Finalno treniranje na celoj istoriji
    DeepNet epoha 10/13
    LightGBM glava 1/7
    LightGBM glava 2/7
    LightGBM glava 3/7
    LightGBM glava 4/7
    LightGBM glava 5/7
    LightGBM glava 6/7
    LightGBM glava 7/7

FINALNI NEXT: 1, 8, 9, 11, 23, 32, 37
Zajednički modelski skor: 4.94326618004e-07
Posećenih čvorova pretrage: 9
Najduži uzastopni niz: 2
Brojeva 1..19 / 20..39: 4 / 3
Ponderi: DeepNet=0.50, LightGBM=0.35, Bayes=0.15
Modelski skor je nenormalizovan zajednički skor za rangiranje legalnih kombinacija.



============================================================================
loto7_4704_k83_loto_2974.csv
Izvlačenja: 2974
Istorijskih prelaza za finalno treniranje: 2942
SHA-256: fc5313380978e4826789413029140d99f09bd92c143892bf9b9027ff9f276be7
Validacija: CSV redovi 2863..2910
Završna provera: CSV redovi 2911..2974

1. Izbor dužine treniranja na validaciji
    DeepNet epoha 10/40
    LightGBM glava 1/7
    LightGBM glava 2/7
    LightGBM glava 3/7
    LightGBM glava 4/7
    LightGBM glava 5/7
    LightGBM glava 6/7
    LightGBM glava 7/7
  Izabrano DeepNet epoha: 11
  LightGBM iteracije po regresoru: 1..41

2. Treniranje pre odvojene završne provere
    DeepNet epoha 10/11
    LightGBM glava 1/7
    LightGBM glava 2/7
    LightGBM glava 3/7
    LightGBM glava 4/7
    LightGBM glava 5/7
    LightGBM glava 6/7
    LightGBM glava 7/7
  Provera 16/64; čvorova poslednje pretrage: 6
  Provera 32/64; čvorova poslednje pretrage: 51
  Provera 48/64; čvorova poslednje pretrage: 71
  Provera 64/64; čvorova poslednje pretrage: 8

REZULTATI ODVOJENE PROVERE
Prosečno pogodaka: 1.4844
Raspodela pogodaka 0..7: [6, 29, 22, 6, 1, 0, 0, 0]
Kola sa 3+: 7
Kola sa 4+: 1
Kontrola: prethodno izvuceno kolo — 1.3125
Razlika prema kontroli: +0.1719
Različite prognoze: 57/64
Prognoze sa 5+ uzastopnih brojeva: 0
Prognoze potpuno u jednoj polovini: 0

3. Finalno treniranje na celoj istoriji
    DeepNet epoha 10/11
    LightGBM glava 1/7
    LightGBM glava 2/7
    LightGBM glava 3/7
    LightGBM glava 4/7
    LightGBM glava 5/7
    LightGBM glava 6/7
    LightGBM glava 7/7

FINALNI NEXT: 2, 8, 13, 18, 25, 30, 39
Zajednički modelski skor: 3.74618193727e-07
Posećenih čvorova pretrage: 7
Najduži uzastopni niz: 1
Brojeva 1..19 / 20..39: 4 / 3
Ponderi: DeepNet=0.50, LightGBM=0.35, Bayes=0.15
Modelski skor je nenormalizovan zajednički skor za rangiranje legalnih kombinacija.



============================================================================
loto7_4704_k83_loto_plus_1730.csv
Izvlačenja: 1730
Istorijskih prelaza za finalno treniranje: 1698
SHA-256: 1f4252375372cbbb88cd026892dc19eb5400e6386bfc5dc613ff7d19d9db0504
Validacija: CSV redovi 1619..1666
Završna provera: CSV redovi 1667..1730

1. Izbor dužine treniranja na validaciji
    DeepNet epoha 10/40
    DeepNet epoha 20/40
    LightGBM glava 1/7
    LightGBM glava 2/7
    LightGBM glava 3/7
    LightGBM glava 4/7
    LightGBM glava 5/7
    LightGBM glava 6/7
    LightGBM glava 7/7
  Izabrano DeepNet epoha: 13
  LightGBM iteracije po regresoru: 1..59

2. Treniranje pre odvojene završne provere
    DeepNet epoha 10/13
    LightGBM glava 1/7
    LightGBM glava 2/7
    LightGBM glava 3/7
    LightGBM glava 4/7
    LightGBM glava 5/7
    LightGBM glava 6/7
    LightGBM glava 7/7
  Provera 16/64; čvorova poslednje pretrage: 17
  Provera 32/64; čvorova poslednje pretrage: 40
  Provera 48/64; čvorova poslednje pretrage: 22
  Provera 64/64; čvorova poslednje pretrage: 88

REZULTATI ODVOJENE PROVERE
Prosečno pogodaka: 1.2812
Raspodela pogodaka 0..7: [12, 26, 22, 4, 0, 0, 0, 0]
Kola sa 3+: 4
Kola sa 4+: 0
Kontrola: prethodno izvuceno kolo — 1.2812
Razlika prema kontroli: +0.0000
Različite prognoze: 62/64
Prognoze sa 5+ uzastopnih brojeva: 0
Prognoze potpuno u jednoj polovini: 0

3. Finalno treniranje na celoj istoriji
    DeepNet epoha 10/13
    LightGBM glava 1/7
    LightGBM glava 2/7
    LightGBM glava 3/7
    LightGBM glava 4/7
    LightGBM glava 5/7
    LightGBM glava 6/7
    LightGBM glava 7/7

FINALNI NEXT: 2, 5, 10, 23, 27, 29, 39
Zajednički modelski skor: 1.92781180899e-07
Posećenih čvorova pretrage: 39
Najduži uzastopni niz: 1
Brojeva 1..19 / 20..39: 3 / 4
Ponderi: DeepNet=0.50, LightGBM=0.35, Bayes=0.15
Modelski skor je nenormalizovan zajednički skor za rangiranje legalnih kombinacija.


============================================================================
ZAVRŠNI PREGLED
loto7_4704_k83.csv: 1, 8, 9, 11, 23, 32, 37
loto7_4704_k83_loto_2974.csv: 2, 8, 13, 18, 25, 30, 39
loto7_4704_k83_loto_plus_1730.csv: 2, 5, 10, 23, 27, 29, 39
"""



"""
zajednička mreža sa više izlaznih glava, LightGBM sa vremenskim ponderisanjem i Bayesova komponenta, spojeni u ansambl 

- Učitavanje tri cela CSV-a i zasebno treniranje za svaki.
- Prilagođavanje četiri cifarske glave zajedničkom izboru sedam različitih brojeva 1-39.- Izostavljanje kalendarskih i numeroloških dodataka, jer naši CSV-ovi nemaju te podatke.

Ansambl koristi fiksne pondere 50/35/15, a provera jednu hronološku podelu 75/25. 
Tokom te provere kontekst ostaje zamrznut na kraju treninga — taj deo za svako naredno kolo koristi tada dostupnu istoriju.


tri grane projekta i ponderi 50/35/15 
Neuronska mreža i LightGBM rade regresiju distribucije, a Bayesova grana uslovnu empirijsku distribuciju. 
Završna kombinacija bira se zajednički, bez zaokruživanja sedam regresionih izlaza i bez filtera za raspored brojeva.


Nema slučajne inicijalizacije, dropout-a, mešanja uzoraka ni nasumičnog uzorkovanja osobina. 
LightGBM podešavanja proverena su prema zvaničnoj dokumentaciji.


Potrebne biblioteke: 
python3 -m pip install numpy torch lightgbm


Ako je SciPy instaliran, ali njegov binarni modul ne može da se učita na macOS-u. 
Ovaj problem je prijavljen SciPy projektu; novija grana koristi prepisani PROPACK modul.

U terminalu reinstaliraj samo SciPy u istom okruženju, bez menjanja ostalih biblioteka:
python -m pip install --force-reinstall --no-cache-dir --only-binary=:all: --no-deps "scipy==1.17.1"

Zatim proveri učitavanje i zavisnosti:
/Users/4c/tesla_env/bin/python -m pip check

/Users/4c/tesla_env/bin/python -c "import scipy.sparse.linalg; import torch; import lightgbm; print('Učitavanje biblioteka: OK')"
Ako provere prođu ---> OK
"""
