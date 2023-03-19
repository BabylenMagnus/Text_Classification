import sys
import requests


if __name__ == '__main__':
    url = sys.argv[1]
    while True:
        a = input().replace(" ", "+")
        out = requests.get(url + "/?" + a)
        print(out.text)
