
TEST_CASE_IPYNB = Tube_checks.ipynb Tube\ closing.ipynb Tube\ opening.ipynb
TEST_CASE_PDF = Tube_checks.pdf Tube\ closing.pdf Tube\ opening.pdf
IPYNB2PDF = jupyter nbconvert --to pdf --execute

%.pdf: %.ipynb
	$(eval ipynbtemp := $(<:.ipynb=_tmp.ipynb))
	sed -e 's/%matplotlib\ \+notebook/%matplotlib inline/g' $< >  $(ipynbtemp)
	$(IPYNB2PDF) $(ipynbtemp) --output=$@

test-cases.pdf: $(TEST_CASE_PDF)
	pdftk $(TEST_CASE_PDF) cat output test-case.pdf

report: test-cases.pdf

