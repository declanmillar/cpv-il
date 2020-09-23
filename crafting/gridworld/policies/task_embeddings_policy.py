import torch
from gridworld.algorithms.composite_dataset import ActionToTensor


class LearnedTECNetPolicy:
    """

    """

    def __init__(
        self,
        state_dict,
        device,
        model=None,
        agent_centric=False,
        env_dim=(63, 60, 3),
        task_embedding_dim=256,
        relu=False,
    ):
        print("model", model, task_embedding_dim)
        self.model = model(task_embedding_dim=task_embedding_dim).to(device)
        self.model.load_state_dict(state_dict)
        self.agent_centric = agent_centric
        self.model.eval()
        self.transformer = ActionToTensor()
        self.device = device
        self.env_dim = env_dim

    def get_goal_feat(self, img_pre, img_post):
        img_pre = self.transformer.convert_image(img_pre).unsqueeze(0).to(self.device)
        img_post = self.transformer.convert_image(img_post).unsqueeze(0).to(self.device)
        goal_feat = self.model.get_goal_feat(img_pre, img_post)
        return goal_feat

    def get_action_from_ref(self, img, first_image, goal_feat, return_delta=False):
        with torch.no_grad():
            if self.agent_centric:
                img = img.reshape(self.env_dim)  # *255
            else:
                img = img.reshape(self.env_dim)  # *255
            image = self.transformer.convert_image(img).unsqueeze(0).to(self.device)
            img_first = self.transformer.convert_image(first_image).unsqueeze(0).to(self.device)
            logits = self.model.forward(img_first, image, goal_feat=goal_feat)[0].squeeze(0).squeeze(0)
            # logits = self.model.forward(img_first, image,)
            # logits = logits.squeeze(0).squeeze(0)
            M = torch.distributions.categorical.Categorical(logits=logits)
            action = M.sample().item()
        if return_delta:
            return action, delta
        return action

    def get_action(self, img, img_pre, img_post, first_image=None):
        with torch.no_grad():
            if self.agent_centric:
                img = img.reshape(self.env_dim)  # *255
            else:
                img = img.reshape(self.env_dim)  # *255
            # import pdb; pdb.set_trace()
            image = self.transformer.convert_image(img).unsqueeze(0).to(self.device)
            img_pre = self.transformer.convert_image(img_pre).unsqueeze(0).to(self.device)
            img_post = self.transformer.convert_image(img_post).unsqueeze(0).to(self.device)
            if first_image is not None:
                img_first = self.transformer.convert_image(first_image).unsqueeze(0).to(self.device)
                logits = self.model.forward(img_first, image, img_pre, img_post)[0].squeeze(0).squeeze(0)
            else:
                logits = self.model.forward(image, img_pre, img_post)[0].squeeze(0).squeeze(0)
            M = torch.distributions.categorical.Categorical(logits=logits)
            action = M.sample().item()
        return action

