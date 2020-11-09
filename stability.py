#!/usr/bin/python3
# Description -----------------------------------------------------------------
# Compute the relative affinity from a "bound" population and an "unbound"
# population.
# -----------------------------------------------------------------------------

import argparse
from numpy import log, exp
from os.path import realpath, dirname
from random import choice

def parse_arguments():
    "Parsing command line"
    parser = argparse.ArgumentParser(description="Compute the relative stability using biased sequences")
    parser.add_argument('pop', help="population")
    parser.add_argument('bias', help="bias values")
    parser.add_argument('-e', '--eref', help="reference energies")
    parser.add_argument('-rf', '--ref_seq', help = "reference sequence")
    parser.add_argument('-kT', default=0.6, help="temperature (kT)", type=float)
    parser.add_argument('-kTo', default=0.6, help="temperature (kT) for output proba", type=float)
    parser.add_argument('--thres', default=100, help="minimum count", type=int)
    parser.add_argument('-p', '--positions', nargs="*", help = "active positions", required=True)
    return parser.parse_args()

def read_aa_dict(infile=dirname(realpath(__file__))+"/../lib/amino_acids.info"):
    "read amino dict"
    results = {}
    for line in open(infile):
        val = line.strip().split()
        results[val[0]] = val[1]
        results[val[1]] = val[0]
    return results

def compute_bias_ener(seq, bias, positions):
    acc = 0.
    for i, (posi, aai) in enumerate(zip(positions, seq)):
        for posj, aaj in list(zip(positions, seq))[i:]:
            try:
                acc += bias[(posi, posj, CODES[aai], CODES[aaj])]
            except:
                "not a biased pair"
    return -acc

def compute_eref_ener(seq, eref):
    if eref is None:
        return 0.
    acc = 0.
    for aa in seq:
        try:
            acc += eref[CODES[aa]]
        except:
            print(CODES[aa], "not in eref set")
    return acc

def main():
    args = parse_arguments()
    global CODES
    CODES = read_aa_dict()

    # Get the populations -----------------------------------------------------
    pop = {}
    for line in open(args.pop):
        val = line.strip().split()
        if int(val[4]) >= args.thres:
                pop[val[3]] = float(val[4])

    # Read bias and Eref ------------------------------------------------------
    bias = {}
    for line in open(args.bias):
        if not line.startswith("#"):
            val = line.strip().split()
            posi, aai, posj, aaj, bias_val = tuple(val)
            # increment if multiple bias simulations
            try:
                bias[(posi, posj, aai, aaj)] += float(bias_val)
                if posi != posj:
                    bias[(posj, posi, aaj, aai)] += float(bias_val)
            except KeyError:
                bias[(posi, posj, aai, aaj)] = float(bias_val)
                if posi == posj:
                    bias[(posj, posi, aaj, aai)] = float(bias_val)

    eref = None
    if args.eref:
        eref = {}
        for line in open(args.eref):
            val = line.strip().split()
            if len(val) == 2:
                eref[val[0]] = float(val[1])

    # Reference sequence ------------------------------------------------------
    ref_seq = args.ref_seq if args.ref_seq else choice(pop.keys())
    assert len(ref_seq) == len(args.positions), "wrong number of positions!"
    bias_ref = compute_bias_ener(ref_seq, bias, args.positions)
    eref_ref = compute_eref_ener(ref_seq, eref)
    pop_ref = pop[ref_seq]
    sta_ref = -args.kT * log(pop_ref) - bias_ref - eref_ref

    print("# {:-^77}".format(" Relative Stability "))
    print("# reference:     {}".format(ref_seq))
    print("# population nb: {}".format(len(pop)))
    print("# positions:     {}".format(" ".join(map(str, args.positions))))
    print("#", "-"*77)

    # Compute the relative stability ------------------------------------------
    results = []
    for seq, p_seq in pop.items():
        b_seq = compute_bias_ener(seq, bias, args.positions)
        er_seq = compute_eref_ener(seq, eref)
        # remove bias contribution and add reference state
        sta_seq = -args.kT * log(p_seq) - b_seq - er_seq
        stability = sta_seq - sta_ref
        results += [(seq, stability)]

    # Sort then print ---------------------------------------------------------
    results.sort(key = lambda seq_sta: seq_sta[1])
    proba = [exp(-nrj/args.kTo) for seq, nrj in results]
    Z = sum(proba)
    for (seq, stability), prob in zip(results, proba):
        print("{:s} {:<10.3f} {:<10.3f}".format(seq, stability, prob/Z * 100))

if __name__ == '__main__':
    main()
