
import click
from hamerspace.optimizer import optimize_model

@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--quantize', is_flag=True, help="Apply post-training quantization")
@click.option('--prune', is_flag=True, help="Apply pruning")
@click.option('--output', default='optimized_model', help="Output directory for the optimized model")
def main(model_path, quantize, prune, output):
    """Hamerspace CLI tool for shrinking and optimizing AI models."""
    optimize_model(model_path, quantize, prune, output)

if __name__ == '__main__':
    main()
