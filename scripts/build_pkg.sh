set -e
# set -x 

workdir=$(dirname $0)/../

cd $workdir
version=$(grep '^VERSION' setup.py |awk -F "\"|'" '{print $(NF-1)}')
short_commit_id=$(git rev-parse --short HEAD)
commit_comment=$(git log -1 --pretty=%B)
echo $commit_comment $short_commit_id

sleep 1

[ -d dist ] || mkdir dist
tar zcf dist/aiearth-deeplearning-$version.tar.gz --exclude=aiearth/deeplearning/model_zoo/pretrained setup.py aiearth requirements.txt MANIFEST.in
(cd dist && ln -sf aiearth-deeplearning-$version.tar.gz aiearth-deeplearning.tar.gz )
echo "package finished: dist/aiearth-deeplearning-$version.tar.gz"
