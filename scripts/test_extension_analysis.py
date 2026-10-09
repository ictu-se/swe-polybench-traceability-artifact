"""Guard against pseudoreplication and silent deletion of incomplete matrices."""
import unittest
import pandas as pd
from analyze_extension import require_matrix,contrast

class ExtensionTests(unittest.TestCase):
    def test_complete_means_no_duplicate_or_missing(self):
        expected={('a',11),('a',29)}
        with self.assertRaises(ValueError):require_matrix([{'id':'a','seed':11}]*2,('id','seed'),expected,'test')
        require_matrix([{'id':'a','seed':11},{'id':'a','seed':29}],('id','seed'),expected,'test')

    def test_seed_means_do_not_inflate_sample_size(self):
        rows=[]
        for task in ['a','b']:
            for seed in [11,29,47]:
                for method,value in [('left',1.),('right',0.)]:
                    rows.append(dict(task_id=task,repo=task,seed=seed,method=method,recall_10=value))
        result=contrast(pd.DataFrame(rows),'left','right','test','task_seed_mean')
        self.assertEqual(result['n'],2)
        self.assertEqual(result['difference'],1.)
        self.assertEqual(result['cluster_low'],1.)
        self.assertEqual(result['cluster_high'],1.)

    def test_pairing_uses_task_ids_not_row_order(self):
        f=pd.DataFrame([dict(task_id=t,repo=t,method=m,seed=11,recall_10=v)
                        for t,m,v in [('a','left',1.),('b','right',.2),('b','left',.4),('a','right',.8)]])
        result=contrast(f,'left','right','test')
        self.assertAlmostEqual(result['difference'],.2)
        self.assertEqual(result['wins'],2)

if __name__=='__main__':unittest.main()
