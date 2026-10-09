"""Tests of evidence boundaries and the paired navigation budget."""
import unittest
from unittest.mock import patch
from navigation_agent import RepositoryTools, episode

class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.files=[{'path':f'f{i:03d}.py','text':'generic implementation'} for i in range(50)]
        self.files[-1]['text']='unique_external_feature rare_target_marker'
        self.initial=[f['path'] for f in self.files[:40]]

    def test_search_scope_and_reads(self):
        a=RepositoryTools(self.files,self.initial,'restricted40')
        b=RepositoryTools(self.files,self.initial,'repository')
        action={'action':'search','query':'rare_target_marker'}
        self.assertNotIn('f049.py',[m['path'] for m in a.execute(action)['matches']])
        self.assertEqual(b.execute(action)['matches'][0]['path'],'f049.py')
        self.assertIn('error',a.execute({'action':'read','path':'f049.py'}))
        self.assertEqual(b.execute({'action':'read','path':'f049.py'})['path'],'f049.py')
        self.assertEqual(b.finalize([50]),(True,['f049.py']))

    def test_unseen_duplicates_boolean_rejected(self):
        t=RepositoryTools(self.files,self.initial,'repository')
        for ids in [[50],[1,1],[True],[0],list(range(1,12))]:self.assertEqual(t.finalize(ids),(False,[]))
        self.assertEqual(t.finalize([]),(True,[]))

    def test_fixed_action_budget_and_no_gold(self):
        calls=[]
        def fake(host,prompt,seed,final=False):
            calls.append((prompt,final))
            return {'request':{'prompt':prompt},'response':{'response':'{"ranking":[1]}' if final else
                '{"action":"read","path":"f000.py","start_line":1}'},'elapsed_seconds':0}
        task={'problem_statement':'Implement the requested behavior','patch':'SECRET_GOLD','hints_text':'SECRET_HINT'}
        with patch('navigation_agent.generate',fake):
            r=episode(task,{'rankings':{'hybrid_rrf':self.initial}},self.files,'repository',11,'unused')
        self.assertEqual(len(calls),5);self.assertEqual(r['tool_actions'],4)
        self.assertTrue(r['valid']);self.assertEqual(r['ranking'],['f000.py'])
        for prompt,_ in calls:
            self.assertNotIn('SECRET_GOLD',prompt);self.assertNotIn('SECRET_HINT',prompt)

if __name__=='__main__':unittest.main()
