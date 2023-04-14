grep 'import mmseg' -R aietorch   | awk -F ':' '{print $1}' | xargs sed -i "" "s@import mmseg@import aietorch.engine.mmseg@g"
grep 'from mmseg' -R aietorch   | awk -F ':' '{print $1}' | xargs sed -i "" "s@from mmseg@from aietorch.engine.mmseg@g"
