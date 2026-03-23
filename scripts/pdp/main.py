"""
Main entry point for PDP Analysis.

This module provides the primary interface for running PDP analysis,
either programmatically or via command line.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import PDPConfig, load_config, create_default_config
from .data import load_dataset, apply_buffer_transform, Dataset
from .core import PDPAnalyzer, PDPResult
from .visualizations import (
    HeatmapVisualizer,
    HierarchicalClusteringVisualizer,
    MDSVisualizer,
    TSNEVisualizer,
    TopKVisualizer,
    TrajectoryVisualizer,
    AutoencoderVisualizer,
    InequalityVisualizer,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class PDPRunner:
    """
    Main runner class for PDP analysis.
    
    Orchestrates data loading, analysis, and visualization generation
    for all configured PDP variants.
    """
    
    def __init__(self, config: PDPConfig):
        """
        Initialize runner.
        
        Args:
            config: PDP configuration
        """
        self.config = config
        self.results: dict = {}
        
    def run(self) -> dict:
        """
        Run complete PDP analysis.
        
        Returns:
            Dictionary mapping variant names to PDPResult objects
        """
        start_time = time.time()
        logger.info("Starting PDP Analysis...")
        logger.info(f"Dataset: {self.config.dataset_path}")
        logger.info(f"Output folder: {self.config.output_folder}")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset(self.config.dataset_path)
        logger.info(
            f"Loaded: {dataset.num_configurations} configurations, "
            f"{dataset.num_timestamps} timestamps, "
            f"{dataset.num_points} points"
        )
        if dataset.has_classes:
            logger.info(f"Classes detected: {dataset.get_unique_classes()}")
        
        # Update config with dataset dimensions
        self.config.num_configurations = dataset.num_configurations
        self.config.num_timestamps = dataset.num_timestamps
        self.config.num_points = dataset.num_points
        self.config.has_classes = dataset.has_classes
        
        # Generate trajectory visualizations (before PDP analysis)
        if self.config.visualizations.static_absolute:
            self._generate_trajectories(dataset, "absolute")
        if self.config.visualizations.static_relative:
            self._generate_trajectories(dataset, "relative")
        if self.config.visualizations.static_finetuned:
            self._generate_trajectories(dataset, "finetuned")
        
        # Run each enabled PDP variant
        variants_to_run = self.config.get_active_variants()
        
        for variant in variants_to_run:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running PDP variant: {variant}")
            logger.info('='*50)
            
            result = self._run_variant(variant, dataset)
            self.results[variant] = result
            
            # Generate visualizations for this variant
            self._generate_visualizations(result, dataset)
            
            # Save distance matrix
            output_path = self.config.get_output_path("distance_matrices")
            matrix_file = output_path / f"N_C_PDPg_{variant}_DistanceMatrix.csv"
            result.save_distance_matrix(str(matrix_file))
            logger.info(f"Saved distance matrix: {matrix_file}")
        
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*50}")
        logger.info(f"PDP Analysis complete! Total time: {elapsed:.2f} seconds")
        logger.info('='*50)
        
        return self.results
    
    def _run_variant(
        self,
        variant: str,
        base_dataset: Dataset
    ) -> PDPResult:
        """Run a single PDP variant."""
        variant_start = time.time()
        
        # Apply transformations based on variant
        if variant == "fundamental":
            dataset = base_dataset
            rough_x, rough_y = 0.0, 0.0
        elif variant == "buffer":
            dataset = apply_buffer_transform(
                base_dataset,
                self.config.buffer_x,
                self.config.buffer_y
            )
            rough_x, rough_y = 0.0, 0.0
        elif variant == "rough":
            dataset = base_dataset
            rough_x = self.config.rough_x
            rough_y = self.config.rough_y
        elif variant == "buffer_rough":
            dataset = apply_buffer_transform(
                base_dataset,
                self.config.buffer_x,
                self.config.buffer_y
            )
            rough_x = self.config.rough_x
            rough_y = self.config.rough_y
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # Run PDP analysis
        analyzer = PDPAnalyzer(
            dataset=dataset,
            window_length=self.config.window_length,
            rough_x=rough_x,
            rough_y=rough_y
        )
        
        result = analyzer.run(variant_name=variant)
        
        elapsed = time.time() - variant_start
        logger.info(f"Variant '{variant}' completed in {elapsed:.2f}s")
        
        if result.identical_groups:
            logger.info(f"Found {len(result.identical_groups)} groups of identical configurations:")
            for group in result.identical_groups:
                logger.info(f"  {group}")
        
        return result
    
    def _generate_trajectories(self, dataset: Dataset, mode: str):
        """Generate trajectory visualizations."""
        logger.info(f"Generating {mode} trajectory visualizations...")
        
        visualizer = TrajectoryVisualizer(self.config, mode=mode)
        
        # Create a dummy result for the variant name
        class DummyResult:
            variant_name = "trajectories"
        
        files = visualizer.visualize(DummyResult(), dataset)
        logger.info(f"Generated {len(files)} trajectory images")
    
    def _generate_visualizations(self, result: PDPResult, dataset: Dataset):
        """Generate all enabled visualizations for a PDP result."""
        viz_config = self.config.visualizations
        
        visualizers = []
        
        if viz_config.heatmap:
            visualizers.append(HeatmapVisualizer(self.config))
        
        if viz_config.hclust:
            visualizers.append(HierarchicalClusteringVisualizer(self.config))
        
        if viz_config.mds:
            visualizers.append(MDSVisualizer(self.config))
        
        if viz_config.tsne:
            visualizers.append(TSNEVisualizer(self.config))
        
        if viz_config.topk:
            visualizers.append(TopKVisualizer(self.config))
        
        if viz_config.autoencoder:
            visualizers.append(AutoencoderVisualizer(self.config))
        
        if viz_config.inequality_matrices:
            visualizers.append(InequalityVisualizer(self.config))
        
        for viz in visualizers:
            logger.info(f"Generating {viz.name} visualization...")
            try:
                files = viz.visualize(result, dataset)
                logger.info(f"  Generated {len(files)} file(s)")
            except Exception as e:
                logger.error(f"  Error generating {viz.name}: {e}")


def run_analysis(
    dataset_path: str,
    output_folder: str,
    config: Optional[PDPConfig] = None,
    **kwargs
) -> dict:
    """
    Convenience function to run PDP analysis.
    
    Args:
        dataset_path: Path to input CSV dataset
        output_folder: Directory for outputs
        config: Optional pre-configured PDPConfig
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary mapping variant names to PDPResult objects
    """
    if config is None:
        config = create_default_config(dataset_path, output_folder)
        
        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    runner = PDPRunner(config)
    return runner.run()


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="PDP (Pairwise Distance Pattern) Analysis for Moving Objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pdp data.csv ./output
  python -m pdp data.csv ./output --variants fundamental buffer
  python -m pdp --config config.json
        """
    )
    
    parser.add_argument(
        'dataset',
        nargs='?',
        help='Path to input CSV dataset'
    )
    parser.add_argument(
        'output',
        nargs='?',
        help='Output directory'
    )
    parser.add_argument(
        '--config', '-c',
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--variants',
        nargs='+',
        choices=['fundamental', 'buffer', 'rough', 'buffer_rough'],
        default=['fundamental'],
        help='PDP variants to run (default: fundamental)'
    )
    parser.add_argument(
        '--window-length', '-w',
        type=int,
        default=3,
        help='Sliding window length (default: 3)'
    )
    parser.add_argument(
        '--no-heatmap',
        action='store_true',
        help='Disable heatmap visualization'
    )
    parser.add_argument(
        '--no-mds',
        action='store_true',
        help='Disable MDS visualization'
    )
    parser.add_argument(
        '--no-tsne',
        action='store_true',
        help='Disable t-SNE visualization'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load or create config
    if args.config:
        config = load_config(args.config)
    elif args.dataset and args.output:
        from .config import VisualizationConfig
        
        viz_config = VisualizationConfig(
            heatmap=not args.no_heatmap,
            mds=not args.no_mds,
            tsne=not args.no_tsne,
        )
        
        config = PDPConfig(
            dataset_path=args.dataset,
            output_folder=args.output,
            window_length=args.window_length,
            visualizations=viz_config,
            run_fundamental='fundamental' in args.variants,
            run_buffer='buffer' in args.variants,
            run_rough='rough' in args.variants,
            run_buffer_rough='buffer_rough' in args.variants,
        )
    else:
        parser.print_help()
        sys.exit(1)
    
    # Run analysis
    try:
        results = run_analysis(
            config.dataset_path,
            config.output_folder,
            config=config
        )
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {config.output_folder}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
