import argparse
import subprocess


def query(dataset, primary_func=None, proc_func=None, datatier_func=None):
    """Return files for given DAS query via dasgoclient"""

    try:
        _, primary, proc, datatier = dataset.split('/')
        if primary_func is not None:
            primary = primary_func(primary)
        if proc_func is not None:
            proc = proc_func(proc)
        if datatier_func is not None:
            datatier = datatier_func(datatier)
        dataset = '/'.join(['', primary, proc, datatier])
    except Exception as e:
        print('Error: ', e)

    query = 'dataset dataset=%s' % dataset
    cmd = ['dasgoclient', '-query', query]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outs, errs = proc.communicate()
    print(cmd)
    print(outs)
    found = outs.splitlines()
    return found


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess ntuples')
    parser.add_argument('-i', '--input', required=True, help='Input config file.')
    parser.add_argument('-o', '--output', required=True, help='Output config file.')
    parser.add_argument('--proc', required=True, help='Replacement process string.')
    args = parser.parse_args()

    with open(args.output, 'w') as fout:
        with open(args.input) as fin:
            for line in fin:
                line = line.strip()
                if '/' not in line[:5]:
                    fout.write(line + '\n')
                    continue
                prefix = ''
                if line.startswith('#'):
                    line = line.replace('#', '').strip()
                    prefix = '# '
                found = query(line, proc_func=lambda x: args.proc)
                for ds in found:
                    fout.write(prefix + ds + '\n')
