workdir=$(dirname $0)/../

cd $workdir
version=$(grep '^VERSION' setup.py |awk -F "\"|'" '{print $(NF-1)}')
ProgramCommitID=$(git rev-parse HEAD)
echo ${ProgramCommitID} > aietorch/COMMIT_ID

mkdir dist
tar zcvf dist/aie-aietorch-$version.tar.gz setup.py aietorch requirements.txt MANIFEST.in
(cd dist && ln -sf aie-aietorch-$version.tar.gz aie-aietorch.tar.gz )
echo "package finished: dist/aie-aietorch-$version.tar.gz"
rm -f aietorch/COMMIT_ID
