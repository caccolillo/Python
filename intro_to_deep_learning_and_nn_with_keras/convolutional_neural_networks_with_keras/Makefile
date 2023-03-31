#to install the virtual environment, use: pip3 install virtualenv
.env:
	virtualenv .env

.PHONY: deps
deps: .env
	. .env/bin/activate; \
	pip install -r requirements.txt
	
.PHONY: run
run:
	. .env/bin/activate; \
	./run_script_py
	
.PHONY: distclean
distclean:
	rm -rf .env
	
.PHONY: vscode
vscode:
	code .
