batch=1 #309
seqlen=256
kNumHead=2
vNumHead=2
kHeadDim=128
vHeadDim=128
isVariedLen=0 #1
chunkSize=64
dtype="bf16"
useActualInput=0
useActualOutput=0
device=2
gDType="float"
stateDType="float"

useInitialState=0
storeFinalState=0
dataPath="/path/to/data"

echo 'Case: batch=' $batch ' seqlen=' $seqlen ' kNumHead=' $kNumHead  ' vNumHead=' $vNumHead ' kHeadDim=' $kHeadDim ' vHeadDim=' $vHeadDim ' isVariedLen=' $isVariedLen ' chunkSize=' $chunkSize ' dtype=' $dtype
python3 test_fwd_h.py $batch $seqlen $kNumHead $vNumHead $kHeadDim $vHeadDim $isVariedLen $chunkSize $useInitialState $storeFinalState "$dtype" $useActualInput $useActualOutput $dataPath $device $gDType $stateDType
# 真实 GDN 输入（w/u 由 k/v/beta/g 按 chunkSize 做 WY 变换）时，追加第 18 个参数 1：
# python3 test_fwd_h.py $batch $seqlen $kNumHead $vNumHead $kHeadDim $vHeadDim $isVariedLen $chunkSize $useInitialState $storeFinalState "$dtype" $useActualInput $useActualOutput $dataPath $device $gDType $stateDType 1
# 跨 chunk 粒度交叉验证（32 vs 64，同源 WY 输入）：
# python3 compare_chunk_sizes.py --T 640 --K 128 --V 128 --B 1 --HK 1 --HV 1 --device $device
python data_compare_h.py $dtype