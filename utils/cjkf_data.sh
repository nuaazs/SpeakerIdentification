unset j;
set j=0;
for i in *;
do let j+=1;
mv "$i" ./id"$j";
done;