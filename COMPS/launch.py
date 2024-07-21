import os
import sys
import numpy as np
from idmtools.assets import AssetCollection, Asset
from idmtools.core.platform_factory import Platform
from idmtools.entities import CommandLine
from idmtools.builders import SimulationBuilder
from idmtools.entities.experiment import Experiment
from idmtools.entities.templated_simulation import TemplatedSimulations
from idmtools_platform_comps.utils.scheduling import add_schedule_config
from idmtools.entities.command_task import CommandTask

def update_parameter_callback(simulation, walker):
    """This function updates the parmeter values for each individual simulation."""
    ret_tags_dict = {"walker": walker}
    return ret_tags_dict

if __name__ == "__main__":
    # Create a platform to run the workitem
    with Platform("CALCULON", priority="Normal") as platform:
        # create command line input for the task
        cmdline = (
            "singularity exec Assets/numba_on_comps_0.0.2_07b5e79.sif python script.py"
        )

        command = CommandLine(cmdline)
        task = CommandTask(command=command)

        # Add our image
        task.common_assets.add_assets(AssetCollection.from_id_file("sif.id"))

        # Add scripts
        task.transient_assets.add_or_replace_asset(Asset(filename="script.py"))

        ts = TemplatedSimulations(base_task=task)

        sb = SimulationBuilder()
        sb.add_multiple_parameter_sweep_definition(
            update_parameter_callback,
            walker=np.arange(3).tolist(),
        )
        ts.add_builder(sb)
        num_threads = 1
        add_schedule_config(
            ts,
            command=cmdline,
            NumNodes=1,
            num_cores=num_threads,
            node_group_name="idm_abcd",
            Environment={"OMP_NUM_THREADS": str(num_threads)},
        )
        experiment = Experiment.from_template(ts, name=f"numby_on_comps")
        experiment.run(wait_until_done=True, scheduling=True)