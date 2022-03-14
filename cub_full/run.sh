file=$(find cub)
echo $file
for i in $file;do
	sed -i "s/cub::DivideAndRoundUp/CUB_NS_QUALIFIER::DivideAndRoundUp/g" $i
done;
