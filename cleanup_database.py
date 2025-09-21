import sqlite3
from pathlib import Path

def cleanup_database():
    """Remove database records for files that no longer exist"""
    
    project_root = Path(__file__).parent
    db_path = project_root / "data" / "research_agent.db"
    uploads_dir = project_root / "data" / "uploads"
    
    if not db_path.exists():
        print("❌ Database not found")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🧹 CLEANING DATABASE...")
        
        # Get all documents from database
        cursor.execute("SELECT id, filename FROM documents")
        all_docs = cursor.fetchall()
        
        print(f"📋 Found {len(all_docs)} records in database")
        
        # Check which files actually exist
        removed_count = 0
        for doc_id, filename in all_docs:
            # Check for original file
            original_file = uploads_dir / filename
            
            # Check for content file
            content_file = uploads_dir / f"{filename}_content.txt"
            base_name = Path(filename).stem
            alt_content_file = uploads_dir / f"{base_name}_content.txt"
            
            # If no files exist, remove from database
            files_exist = any([
                original_file.exists(),
                content_file.exists(), 
                alt_content_file.exists()
            ])
            
            if not files_exist:
                print(f"🗑️ Removing record: {filename} (ID: {doc_id})")
                
                # Delete from both tables
                cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
                removed_count += 1
            else:
                print(f"✅ Keeping record: {filename}")
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ DATABASE CLEANUP COMPLETE!")
        print(f"📊 Removed {removed_count} invalid records")
        print(f"📊 Remaining records: {len(all_docs) - removed_count}")
        
        # Show current files in uploads
        actual_files = list(uploads_dir.glob("*"))
        actual_files = [f for f in actual_files if f.name != '.gitkeep']
        
        print(f"\n📁 ACTUAL FILES IN UPLOADS: {len(actual_files)}")
        for f in actual_files:
            print(f"   📄 {f.name}")
            
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

if __name__ == "__main__":
    cleanup_database()