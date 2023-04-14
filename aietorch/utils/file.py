import os
import shutil
import zipfile
from fnmatch import fnmatch


def list_dir(path, suffix=None, abspath=False):
    return list_dir_sorted_by_mtime(path, suffix, abspath)


def list_dir_sorted_by_mtime(path, suffix=None, abspath=False):
    path = os.path.abspath(path)
    files = os.listdir(path)
    file_dict = {}
    for f in files:
       f_path = os.path.join(path, f)
       if abspath:
          f = f_path   # os.path.join(path, i)
       if suffix:
          if f.endswith(suffix):
              file_dict[f] = os.path.getmtime(f_path)
       else:
           file_dict[f] = os.path.getmtime(f_path)

    file_order=sorted(file_dict.items(),key=lambda x:x[1],reverse=True)
    ret = [x[0] for x in file_order]
    return ret


def mkdir(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def cp(src, dst):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    shutil.copyfile(src, dst)
    return dst


def make_softlink(src, dst):
    try:
        #print(src, dst)
        os.symlink(src, dst)
    except FileExistsError as e:
        pass


def is_text(content):
    if b'\0' in content:
        return False

    text_characters = ''.join(list(map(chr, range(32, 127))) + list('\n\r\t\b'))
    _null_trans = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff'

    t = content.translate(_null_trans, text_characters.encode())
    if float(len(t))/float(len(content)) > 0.30:
        return False
    else:
        return True


def is_text_file(file_path):
    with open(file_path) as f:
        content = f.read(2048)
        return is_text(content)


def is_empty_file(file_path):
    size = os.path.getsize(file_path)
    if size <= 0:
        return True
    return False

def list_dir_r(dir, exclude=[]):
    file_list = []
    if not os.path.isdir(dir):
        return [dir]

    exclude = [exclude] if type(exclude) == str else exclude
    assert type(exclude) in (tuple, list)

    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            match = False
            for rule in exclude:
                if fnmatch(file_path, rule):
                    match = True
                    break
            if not match:
                file_list.append(os.path.join(root, name))
    return file_list


def zip_dir(zip_path, package_dir, relpath=None, exclude=[], show_detail=False):
    assert os.path.isdir(package_dir)
    file_list = list_dir_r(package_dir, exclude)
    f = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    if relpath == None:
        for file in file_list:
            f.write(file)
            if show_detail:
                print("archive: ", file)
    else:
        for file in file_list:
            arcname = os.path.relpath(file, relpath)
            f.write(file, arcname=arcname)
            if show_detail:
                print("archive: ", file)
    f.close()
    return zip_path


if __name__ == "__main__":
    print(list_dir(".", "py", abspath=False))