MIT LICENSE

usage:<br />
start for first learning<br />
main_tf20v2.py --runmode first<br />
<br />
start from checkpoint<br />
main_tf20v2.py --runmode again<br />
<br />
generate picture from checkpoint<br />
main_tf20v2.py --runmode generate<br />
<br />
SAMPLE:64*64pixel face generation<br />
![train_00019000](https://github.com/katsuhiko-matsumoto/dcgan_tf20/assets/9207497/326f9c82-d43e-4033-b3cc-6b7492d4e213)

<br />
<br />
I collected and processed my own datasets, but you can also use the following free datasets.
Usable data sets:<a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" target="_blank">https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html</a>
<br />
<br />
Adjust the following parameters of the model class to the image size<br />
self.input_width<br />
self.input_height<br />



