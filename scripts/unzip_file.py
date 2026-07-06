import sys

import pyzipper

def main():
    num_tries = 0
    while (num_tries < 3):
        try:
            pwd = input("Please type the password to unzip the downloaded file:\n")
            with pyzipper.AESZipFile("36728994") as zf:
                zf.extractall(pwd=bytes(pwd, "utf-8"))

        except:
            print("\nThe indicated password is wrong, try again.")
            num_tries += 1

        else:
            print("Extraction successful.")
            break

    else:
        print("Too many failed attempts. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
