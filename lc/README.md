# How to run on LC
## Install and Test

We should run dsr in a terminal. On Pascal, we can create one like:

	mxterm -g 160x50+0+0 1 1 1439 -A datasci

Next, set environment using the provided script:
	
	source lc/profile.toss_3_x86_64_ib.dsr

You will need to run this at least once whenever you login.

Next, install tensorflow into a virtual environment. You **MUST** use python 3.6.4 with Tensorflow 1.14.  
	
	/usr/apps/python-3.6.4/bin/virtualenv --system-site-package venv3-tf-cpu
	source venv3-tf-cpu/bin/activate
	
Next, make sure all packages are up-to-date. LC has some really old ones installed and they can cause problems.

	pip install --upgrade pandas
	pip install --upgrade cython
	pip install --upgrade numba
	pip install --upgrade six
	pip install --upgrade h5py

Or you can try and update **EVERYTHING** using:

	pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

Next install Tensorflow and test it.

	pip install tensorflow==1.14
	python -c 'import tensorflow as tf; hello = tf.constant("hello"); sess = tf.Session(); print(sess.run(hello))'
	
Next install and run as usual:

	pip install -r requirements.txt
	git clone https://github.com/trevorstephens/gplearn.git # Clone gplearn
	pip install ./gplearn # Install gplearn

We should now be able to run the source code. Try:

	python -m dsr.run dsr/dsr/config.json --b=Nguyen-1
	
You should see an output that looks like, but can be a a little different from:

	Setting 'num_cores' to 1 for batch because there are only 1 expressions.	
	Running dsr for n=1 on benchmarks ['Nguyen-1']
	WARNING:tensorflow:
	The TensorFlow contrib module will not be included in TensorFlow 2.0.
	For more information, please see:
	  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
	  * https://github.com/tensorflow/addons
	  * https://github.com/tensorflow/io (for I/O related ops)
	If you depend on functionality not listed there, please file an issue.
	
	Library:
	        x1, add, sub, mul, div, sin, cos, exp, log
	
	New best overall
	        Reward: -0.0906692226522936
	        Base reward: -0.0906692226522936
	        Count: 1
	        Traversal: sub,add,add,mul,mul,cos,div,div,x1,add,log,add,log,sub,sub,sub,log,log,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1
	        Expression:

etc....

## Creating an LC Job

