## How to contribute code

Follow these steps to submit your code contribution.

### Step 1. Open an issue

Before making any changes, we recommend opening an issue (if one doesn't already
exist) and discussing your proposed changes. This way, we can give you feedback
and validate the proposed changes.

If the changes are minor (simple bug fix or documentation fix), then feel free
to open a PR without discussion.

### Step 2. Make code changes

To make code changes, you need to fork the repository. You will need to setup a
development environment and run the unit tests. This is covered in section
"Setup environment".

### Step 3. Create a pull request

Once the change is ready, open a pull request from your branch in your fork to
the master branch in [keras-team/keras](https://github.com/keras-team/keras).

### Step 4. Sign the Contributor License Agreement

After creating the pull request, the `google-cla` bot will comment on your pull
request with instructions on signing the Contributor License Agreement (CLA) if
you haven't done so. Please follow the instructions to sign the CLA. A `cla:yes`
tag is then added to the pull request.

![Tag added](https://i.imgur.com/LHEdIfL.png)


### Step 5. Code review

CI tests will automatically be run directly on your pull request.  Their
status will be reported back via GitHub actions.

There may be
several rounds of comments and code changes before the pull request gets
approved by the reviewer.

![Approval from reviewer](https://i.imgur.com/zgRziTt.png)

### Step 6. Merging

Once the pull request is approved, a team member will take care of merging.

## Setup environment

Setting up your KerasCV development environment is quite easy.  You simply
need to run the following commands:

```shell
git clone https://github.com/YOUR_GITHUB_USERNAME/keras-cv.git
cd keras-cv
pip install ".[tests]"
python setup.py develop
```

Following these commands you should be able to run the tests using `pytest keras_cv`.
Please report any issues running tests following these steps.

## Run tests

KerasCV is tested using [PyTest](https://docs.pytest.org/en/6.2.x/).

### Run a test file

To run a test file, run `pytest path/to/file` from the root directory of keras\_cv.

### Run a single test case

To run a single test, you can use `-k=<your_regex>`
to use regular expression to match the test you want to run. For example, you
can use the following command to run all the tests in `cut_mix_test.py`,
whose names contain `label`,

```
pytest keras_cv/layers/preprocessing/cut_mix_test.py -k="label"
```

### Run all tests

Running all of the tests in KerasCV is as easy as:
```
pytest keras_cv/
```

### Formatting the Code
In order to format the code you can use the `shell/format.sh` script.
If this does not resolve the issue, try updating `isort` and `black`
via `pip install --upgrade black` and `pip install --upgrade isort`.