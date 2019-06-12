"""
For the package's command line commands.
"""
import sys
import dreamwarrior

def main():
    """Main function for parsing command line arguments.
    """
    argument = sys.argv[1]

    if argument[-4:] == '.bk2':
        dreamwarrior.play_movie(argument)

if __name__ == '__main__':
    main()
