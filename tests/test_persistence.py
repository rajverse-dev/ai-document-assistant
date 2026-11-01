
from src.vector_store import VectorStore


def main():
    # Reopen the existing local database
    vector_store = VectorStore()

    total = vector_store.count()

    print("Database reopened successfully!")
    print("Stored chunks:", total)

    assert total > 0, "No records found in database"

    print("Persistence test passed!")


if __name__ == "__main__":
    main()
