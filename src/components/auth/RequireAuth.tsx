/**
 * RequireAuth - Guards protected routes
 * Redirects to /login if not authenticated
 */

import { useEffect, type ReactNode } from 'react';
import { useNavigate, Outlet } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';

interface RequireAuthProps {
  children?: ReactNode;
}

export const RequireAuth = ({ children }: RequireAuthProps) => {
  const { isLoggedIn } = useAuthStore();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isLoggedIn) {
      navigate('/login', { replace: true });
    }
  }, [isLoggedIn, navigate]);

  if (!isLoggedIn) return null;
  
  // Support both direct children (for testing) and Outlet (for nested routes)
  return children ? <>{children}</> : <Outlet />;
};
