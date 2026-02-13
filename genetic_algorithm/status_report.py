#!/usr/bin/env python3
"""
Status Report Generator

Generates a comprehensive status report of the genetic algorithm implementation.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def count_lines(file_path):
    """Count lines of code in a file, excluding empty lines and comments."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                comment_lines += 1
            else:
                code_lines += 1
        
        return code_lines, comment_lines, blank_lines
    except Exception:
        return 0, 0, 0


def analyze_directory(directory):
    """Analyze Python files in a directory."""
    total_code = 0
    total_comments = 0
    total_blank = 0
    file_count = 0
    
    for py_file in Path(directory).rglob("*.py"):
        if '__pycache__' in str(py_file):
            continue
        
        code, comments, blank = count_lines(py_file)
        total_code += code
        total_comments += comments
        total_blank += blank
        file_count += 1
    
    return file_count, total_code, total_comments, total_blank


def main():
    """Generate status report."""
    print("=" * 80)
    print(" " * 20 + "GENETIC ALGORITHM IMPLEMENTATION STATUS")
    print("=" * 80)
    print()
    
    # Analyze code base
    print("📊 CODE BASE ANALYSIS")
    print("-" * 80)
    
    ga_dir = Path("genetic_algorithm")
    
    modules = {
        "Core": ga_dir / "core",
        "Strategies": ga_dir / "strategies",
        "Evaluation": ga_dir / "evaluation",
        "Utils": ga_dir / "utils",
    }
    
    total_files = 0
    total_code = 0
    total_comments = 0
    total_blank = 0
    
    for name, directory in modules.items():
        if directory.exists():
            files, code, comments, blank = analyze_directory(directory)
            total_files += files
            total_code += code
            total_comments += comments
            total_blank += blank
            
            print(f"{name:12} | Files: {files:2} | Code: {code:4} | Comments: {comments:4} | Blank: {blank:4}")
    
    print("-" * 80)
    print(f"{'TOTAL':12} | Files: {total_files:2} | Code: {total_code:4} | Comments: {total_comments:4} | Blank: {total_blank:4}")
    print()
    
    # Feature completion
    print("✅ FEATURE COMPLETION")
    print("-" * 80)
    
    features = {
        "Phase 1: Project Setup": 100,
        "Phase 2: Core GA Framework": 100,
        "  - Strategy Representation": 100,
        "  - Population Management": 100,
        "  - Selection Mechanisms": 100,
        "  - Genetic Operators": 100,
        "  - Evolution Loop": 100,
        "Phase 3: Strategy Generation": 100,
        "  - Indicator Library (9 indicators)": 100,
        "  - Random Strategy Generator": 100,
        "  - Strategy Code Generation": 100,
        "  - Condition Generation": 100,
        "Phase 4: Evaluation System": 40,
        "  - Fitness Function": 100,
        "  - Backtesting Integration": 0,
        "Phase 5: Testing & Validation": 80,
        "  - Component Tests": 100,
        "  - Example Scripts": 100,
        "  - Integration Tests": 0,
    }
    
    completed = 0
    total = 0
    
    for feature, percentage in features.items():
        status = "✓" if percentage == 100 else "○" if percentage > 0 else "✗"
        bar_length = 30
        filled = int(bar_length * percentage / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"{status} {feature:40} [{bar}] {percentage:3}%")
        
        if not feature.startswith("  "):  # Top-level features
            completed += percentage
            total += 100
    
    overall = (completed / total) * 100 if total > 0 else 0
    print("-" * 80)
    print(f"{'OVERALL PROGRESS':40} {overall:.1f}%")
    print()
    
    # Implementation highlights
    print("🌟 IMPLEMENTATION HIGHLIGHTS")
    print("-" * 80)
    highlights = [
        "✓ Complete genetic algorithm framework",
        "✓ 3 selection methods (tournament, roulette, rank-based)",
        "✓ 3 crossover operators (single-point, uniform, component)",
        "✓ 4 mutation types (parameter, indicator, condition, structure)",
        "✓ 9 technical indicators supported",
        "✓ Full strategy code generation to FreqTrade format",
        "✓ Comprehensive configuration system",
        "✓ Test and example scripts",
        "✓ Complete documentation",
    ]
    
    for highlight in highlights:
        print(f"  {highlight}")
    print()
    
    # What's missing
    print("⚠️  REMAINING WORK")
    print("-" * 80)
    remaining = [
        "✗ FreqTrade backtesting integration (main blocker)",
        "✗ Result caching system",
        "✗ Checkpointing for long runs",
        "✗ Database storage for strategies",
        "✗ Visualization plots",
        "✗ Comprehensive unit tests",
    ]
    
    for item in remaining:
        print(f"  {item}")
    print()
    
    # Files generated
    print("📁 KEY FILES")
    print("-" * 80)
    
    key_files = [
        ("Core GA", "genetic_algorithm/core/evolution.py"),
        ("Strategy Gene", "genetic_algorithm/core/strategy_gene.py"),
        ("Generator", "genetic_algorithm/strategies/generator.py"),
        ("Fitness", "genetic_algorithm/evaluation/fitness.py"),
        ("Config", "genetic_algorithm/config/ga_config.yaml"),
        ("Tests", "genetic_algorithm/test_generation.py"),
        ("Example", "genetic_algorithm/example_usage.py"),
        ("README", "genetic_algorithm/README.md"),
        ("TODO", "genetic_algorithm/TODO.md"),
        ("Summary", "genetic_algorithm/IMPLEMENTATION_SUMMARY.md"),
    ]
    
    for name, file_path in key_files:
        path = Path(file_path)
        status = "✓" if path.exists() else "✗"
        size = f"{path.stat().st_size:,} bytes" if path.exists() else "N/A"
        print(f"  {status} {name:20} {file_path:50} ({size})")
    print()
    
    # Next steps
    print("🎯 NEXT STEPS")
    print("-" * 80)
    steps = [
        "1. Implement FreqTrade backtesting command execution",
        "2. Parse backtesting JSON/text output",
        "3. Add result caching to avoid redundant tests",
        "4. Test end-to-end with real market data",
        "5. Add comprehensive unit tests",
        "6. Implement checkpointing system",
    ]
    
    for step in steps:
        print(f"  {step}")
    print()
    
    print("=" * 80)
    print(f"Overall Status: {overall:.1f}% Complete")
    print(f"Estimated remaining work: 6-8 hours")
    print("=" * 80)


if __name__ == '__main__':
    main()
