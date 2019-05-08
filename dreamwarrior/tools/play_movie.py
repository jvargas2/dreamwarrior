"""
Script for calling the play_movie() function. 
"""
import argparse
import dreamwarrior

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', help='the name or path of the .bk2 file to use')
    args = parser.parse_args()
    dreamwarrior.play_movie(args.path)