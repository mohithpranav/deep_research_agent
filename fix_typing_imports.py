#!/usr/bin/env python3
"""Fix missing typing imports across the project"""

import os
from pathlib import Path

def fix_imports():
    """Fix typing imports in all Python files"""
    
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    
    # Files to check and fix
    files_to_fix = [
        src_dir / "core" / "document_manager.py",
        src_dir / "reasoning" / "query_analyzer.py", 
        src_dir / "reasoning" / "response_synthesizer.py",
        src_dir / "reasoning" / "multi_step_reasoner.py",
        src_dir / "core" / "embedding_engine.py"
    ]
    
    for file_path in files_to_fix:
        if not file_path.exists():
            continue
            
        try:
            print(f"🔧 Fixing {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check what typing imports are needed
            typing_needed = []
            if 'List[' in content:
                typing_needed.append('List')
            if 'Dict[' in content:
                typing_needed.append('Dict')  
            if 'Any' in content:
                typing_needed.append('Any')
            if 'Optional[' in content:
                typing_needed.append('Optional')
            if 'Union[' in content:
                typing_needed.append('Union')
            if 'Tuple[' in content:
                typing_needed.append('Tuple')
            
            if not typing_needed:
                continue
                
            lines = content.split('\n')
            
            # Find existing typing import line
            typing_line_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('from typing import'):
                    typing_line_idx = i
                    break
            
            if typing_line_idx is not None:
                # Update existing import
                current_line = lines[typing_line_idx]
                current_imports = current_line.split('import')[1].strip().split(',')
                current_imports = [imp.strip() for imp in current_imports]
                
                # Add missing imports
                all_imports = sorted(set(current_imports + typing_needed))
                new_line = f"from typing import {', '.join(all_imports)}"
                lines[typing_line_idx] = new_line
                
            else:
                # Add new typing import after other imports
                import_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ')) and 'typing' not in line:
                        import_idx = i + 1
                
                new_import = f"from typing import {', '.join(sorted(typing_needed))}"
                lines.insert(import_idx, new_import)
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
            print(f"  ✅ Fixed imports: {', '.join(typing_needed)}")
            
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    
    print("✅ All typing imports fixed!")

if __name__ == "__main__":
    fix_imports()
    