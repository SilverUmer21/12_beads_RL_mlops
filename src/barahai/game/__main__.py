from .app import main

if __name__ == "__main__":  # the search worker re-imports this module on spawn
    main()
