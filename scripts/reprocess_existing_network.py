"""
Reprocess existing bipartite network with corrected methodology.

This script takes the old bipartite network and applies the new filtering
methodology (z-scores instead of p-values) and calculates EFC metrics.

Use this if you don't have access to the original PGN data.
"""

import pickle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.project_and_filter_network import project_and_filter_network, save_network

def main():
    """Reprocess existing bipartite network with new methodology."""

    print("=" * 70)
    print("Reprocessing Existing Network with Corrected Methodology")
    print("=" * 70)

    input_file = 'data/bipartite_network.pkl'
    output_file = 'data/relatedness_network.pkl'

    if not os.path.exists(input_file):
        print(f"ERROR: Bipartite network not found at {input_file}")
        print("\nYou need to either:")
        print("1. Have the original bipartite_network.pkl file")
        print("2. Rebuild from PGN data using: python app/build_bipartite_network.py")
        sys.exit(1)

    print(f"\nInput: {input_file}")
    print(f"Output: {output_file}")
    print(f"Z-threshold: 2.0")
    print()

    try:
        # Project and filter with new methodology
        network, metadata = project_and_filter_network(
            bipartite_file=input_file,
            z_threshold=2.0,
            calculate_efc=True
        )

        # Save results
        save_network(network, metadata, output_file)

        print("\n" + "=" * 70)
        print("SUCCESS: Network reprocessed with corrected methodology")
        print("=" * 70)
        print(f"\nNetwork saved to: {output_file}")
        print(f"Metadata saved to: {output_file.replace('.pkl', '_metadata.pkl')}")

        # Print summary
        print("\nNetwork Summary:")
        print(f"  Nodes: {network.number_of_nodes()}")
        print(f"  Edges: {network.number_of_edges()}")
        print(f"  Connected: {metadata['component_analysis']['is_connected']}")

        if not metadata['component_analysis']['is_connected']:
            print(f"  Components: {metadata['component_analysis']['num_components']}")
            print(f"  Largest: {metadata['component_analysis']['largest_component_size']} nodes")

        if 'complexity_stats' in metadata:
            print(f"\nOpening Complexity:")
            print(f"  Mean: {metadata['complexity_stats']['mean']:.3f}")
            print(f"  Std: {metadata['complexity_stats']['std']:.3f}")
            print(f"  Range: [{metadata['complexity_stats']['min']:.3f}, {metadata['complexity_stats']['max']:.3f}]")

        print(f"\nFiltering Stats:")
        print(f"  Original edges: {metadata['filter_stats']['observed_edges']}")
        print(f"  Filtered edges: {metadata['filter_stats']['filtered_edges']}")
        print(f"  Retention rate: {metadata['filter_stats']['retention_rate']:.2%}")

        print("\n" + "=" * 70)
        print("You can now run the Flask app!")
        print("  python app/app.py")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
