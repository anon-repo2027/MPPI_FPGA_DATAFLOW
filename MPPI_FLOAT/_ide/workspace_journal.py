# 2026-02-19T11:35:33.721566
import vitis

client = vitis.create_client()
client.set_workspace(path="MPPI_FLOAT_KINEMATIC")

comp = client.get_component(name="MPPI_Float")
comp.run(operation="SYNTHESIS")

vitis.dispose()

