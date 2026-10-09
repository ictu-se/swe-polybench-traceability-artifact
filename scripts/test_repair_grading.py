import unittest
from repair_probe import grade,parse_tests
class GradingTests(unittest.TestCase):
 def test_base_and_gold(self):
  task={'FAIL_TO_PASS':['tests/a.py::test_new[x]'],'PASS_TO_PASS':['tests/b.py::TestOld::test_ok']}
  base=parse_tests('FAILED tests/a.py::test_new[x] - AssertionError: mismatch\nPASSED tests/b.py::TestOld::test_ok')
  self.assertTrue(grade(base,task,base=True)['passed']);self.assertFalse(grade(base,task)['passed'])
  good=parse_tests('PASSED tests/a.py::test_new[x]\nPASSED tests/b.py::TestOld::test_ok')
  self.assertTrue(grade(good,task)['passed'])
 def test_missing_and_skipped_rejected(self):
  task={'FAIL_TO_PASS':['a'],'PASS_TO_PASS':['b']}
  for outcomes in [{'a':'PASSED'},{'a':'SKIPPED','b':'PASSED'},{'a':'PASSED','b':'XFAIL'}]:
   self.assertFalse(grade(outcomes,task)['passed'])
 def test_upstream_alias_expansion_frozen_before_model(self):
  from repair_probe import expected_test_ids
  task={'FAIL_TO_PASS':['t.py::test[x'],'PASS_TO_PASS':[]}
  mapping=expected_test_ids(task,{'t.py::test[x first]':'PASSED','t.py::test[x second]':'PASSED'})
  self.assertTrue(grade({'t.py::test[x first]':'PASSED','t.py::test[x second]':'PASSED'},task,expected_ids=mapping)['passed'])
  self.assertFalse(grade({'t.py::test[x first]':'PASSED'},task,expected_ids=mapping)['passed'])
  self.assertFalse(grade({'t.py::test[x first]':'PASSED','t.py::test[x second]':'FAILED'},task,expected_ids=mapping)['passed'])
 def test_empty_f2p_not_evidence(self):
  self.assertFalse(grade({'a':'PASSED'},{'FAIL_TO_PASS':[],'PASS_TO_PASS':['a']})['passed'])
if __name__=='__main__':unittest.main()
