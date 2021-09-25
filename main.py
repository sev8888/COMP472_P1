import os

#if __name__ == '__main__':
cwd = os.getcwd()
print(cwd+os.sep+"BBC")
#"""
for subdir, dirs, files in os.walk(cwd):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".txt"):
            print(filepath)
#"""