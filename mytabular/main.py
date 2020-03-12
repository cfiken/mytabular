import argparse
# from mytabular.runner.purchase_together_kfold import PurchaseTogetherRunner
from mytabular.runner.adversarial_validation import AdversarialValidationRunner
from mytabular.runner.purchase_pca_kfold import PurchasePCARunner
from mytabular.runner.purchase_pca_groupkfold import PurchasePCAGroupKFoldRunner


def get_args():
    parser = argparse.ArgumentParser('Pipeline', description='Input configuration for pipeline.')
    parser.add_argument('-n', '--name',
                        type=str,
                        required=True,
                        help='Input name for this running')
    parser.add_argument('-d', '--desc',
                        type=str,
                        help='Input description for this running')
    parser.add_argument('-c', '--config',
                        type=str,
                        help='Input your pipeline config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    name = str(args.name)
    desc = str(args.desc)
    runner = PurchasePCAGroupKFoldRunner(name, desc)
    # runner = PurchasePCARunner(name, desc)
    # runner = AdversarialValidationRunner(name, desc)
    runner.run()
