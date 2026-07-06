import pyzipper

def main():
    pwd = input("Please type the password to unzip the downloaded file:\n")
    with pyzipper.AESZipFile("36728994") as zf:
        zf.extractall(pwd=bytes(pwd, "utf-8"))

if __name__ == "__main__":
    main()
